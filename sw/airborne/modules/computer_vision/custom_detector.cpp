/*
 * custom_detector.cpp
 *
 * Paparazzi color obstacle detector without OpenCV.
 *
 * Detects orange and green obstacles from YUV422 frames and publishes:
 *   CUSTOM_DETECTION(uint8_t left, uint8_t middle, uint8_t right)
 */

extern "C" {
#include "modules/computer_vision/custom_detector.h"
#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"
#include "std.h"
}

#include <pthread.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <chrono>
#include <string>
#include <algorithm>

#define PRINT(string, ...) fprintf(stderr, "[custom_detector->%s()] " string, __FUNCTION__, ##__VA_ARGS__)

#ifndef CUSTOM_DETECT_COLOR_OBJECT_VERBOSE
#define CUSTOM_DETECT_COLOR_OBJECT_VERBOSE TRUE
#endif

#if CUSTOM_DETECT_COLOR_OBJECT_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

#ifndef CUSTOM_DETECT_COLOR_OBJECT_ID
#define CUSTOM_DETECT_COLOR_OBJECT_ID ABI_BROADCAST
#endif

#ifndef CUSTOM_DETECT_COLOR_OBJECT_FPS
#define CUSTOM_DETECT_COLOR_OBJECT_FPS 0
#endif

enum detector_rotation_t {
  DETECTOR_ROTATE_NONE = 0,
  DETECTOR_ROTATE_90_CLOCKWISE = 1,
  DETECTOR_ROTATE_180 = 2,
  DETECTOR_ROTATE_90_COUNTERCLOCKWISE = 3
};

#ifndef CUSTOM_DETECT_COLOR_OBJECT_ROTATION
#define CUSTOM_DETECT_COLOR_OBJECT_ROTATION DETECTOR_ROTATE_90_COUNTERCLOCKWISE
#endif

static pthread_mutex_t detector_mutex;

// HSV-like thresholds (OpenCV scale: H in [0,179], S,V in [0,255])
static const int ORANGE_H_MIN = 5;
static const int ORANGE_H_MAX = 25;
static const int ORANGE_S_MIN = 80;
static const int ORANGE_V_MIN = 120;

static const int GREEN_H_MIN = 25;
static const int GREEN_H_MAX = 90;
static const int GREEN_S_MIN = 40;
static const int GREEN_V_MIN = 40;

// Occupancy thresholds
static const float ORANGE_OCC_THRESHOLD = 0.20f;
static const float GREEN_OCC_THRESHOLD = 0.10f;
static const float WINDOW_LOW_THRESHOLD = 0.10f;

struct custom_detection_result_t {
  uint8_t left;
  uint8_t middle;
  uint8_t right;
  bool updated;
};

static struct custom_detection_result_t global_result;

struct RegionMetrics {
  float orange_occ;
  float green_occ;
  float window_occ;
  bool orange_blocked;
  bool green_blocked;
  bool window_blocked;
  bool blocked;
  int risk_score;
};

static inline uint8_t clamp_u8(int v)
{
  return (uint8_t)std::max(0, std::min(255, v));
}

static void get_processed_dimensions(int src_w, int src_h, int *proc_w, int *proc_h)
{
  if (CUSTOM_DETECT_COLOR_OBJECT_ROTATION == DETECTOR_ROTATE_90_CLOCKWISE ||
      CUSTOM_DETECT_COLOR_OBJECT_ROTATION == DETECTOR_ROTATE_90_COUNTERCLOCKWISE) {
    *proc_w = src_h;
    *proc_h = src_w;
  } else {
    *proc_w = src_w;
    *proc_h = src_h;
  }
}

static void map_processed_to_source(int xr, int yr, int src_w, int src_h, int *xs, int *ys)
{
  switch (CUSTOM_DETECT_COLOR_OBJECT_ROTATION) {
    case DETECTOR_ROTATE_90_COUNTERCLOCKWISE:
      *xs = src_w - 1 - yr;
      *ys = xr;
      break;
    case DETECTOR_ROTATE_90_CLOCKWISE:
      *xs = yr;
      *ys = src_h - 1 - xr;
      break;
    case DETECTOR_ROTATE_180:
      *xs = src_w - 1 - xr;
      *ys = src_h - 1 - yr;
      break;
    case DETECTOR_ROTATE_NONE:
    default:
      *xs = xr;
      *ys = yr;
      break;
  }
}

static bool sample_yuv_uyvy(const struct image_t *img, int x, int y, uint8_t *yy, uint8_t *uu, uint8_t *vv)
{
  if (x < 0 || y < 0 || x >= img->w || y >= img->h) {
    return false;
  }

  int x_even = x & ~1;
  size_t idx = ((size_t)y * (size_t)img->w + (size_t)x_even) * 2u;

  uint8_t u = img->buf[idx + 0u];
  uint8_t y0 = img->buf[idx + 1u];
  uint8_t v = img->buf[idx + 2u];
  uint8_t y1 = img->buf[idx + 3u];

  *uu = u;
  *vv = v;
  *yy = (x == x_even) ? y0 : y1;
  return true;
}

static void yuv_to_hsv(uint8_t y, uint8_t u, uint8_t v, int *h, int *s, int *val)
{
  int c = (int)y - 16;
  int d = (int)u - 128;
  int e = (int)v - 128;

  int r = (298 * c + 409 * e + 128) >> 8;
  int g = (298 * c - 100 * d - 208 * e + 128) >> 8;
  int b = (298 * c + 516 * d + 128) >> 8;

  int rr = clamp_u8(r);
  int gg = clamp_u8(g);
  int bb = clamp_u8(b);

  int cmax = std::max(rr, std::max(gg, bb));
  int cmin = std::min(rr, std::min(gg, bb));
  int delta = cmax - cmin;

  int hue = 0;
  if (delta != 0) {
    if (cmax == rr) {
      hue = 60 * (gg - bb) / delta;
      if (hue < 0) {
        hue += 360;
      }
    } else if (cmax == gg) {
      hue = 60 * (bb - rr) / delta + 120;
    } else {
      hue = 60 * (rr - gg) / delta + 240;
    }
  }

  int sat = (cmax == 0) ? 0 : (255 * delta) / cmax;

  *h = hue / 2;
  *s = sat;
  *val = cmax;
}

static bool is_orange(int h, int s, int v)
{
  return (h >= ORANGE_H_MIN && h <= ORANGE_H_MAX && s >= ORANGE_S_MIN && v >= ORANGE_V_MIN);
}

static bool is_green(int h, int s, int v)
{
  return (h >= GREEN_H_MIN && h <= GREEN_H_MAX && s >= GREEN_S_MIN && v >= GREEN_V_MIN);
}

static void compute_column_metrics_from_yuv(const struct image_t *img,
                                            RegionMetrics *left_m,
                                            RegionMetrics *center_m,
                                            RegionMetrics *right_m)
{
  int proc_w = 0;
  int proc_h = 0;
  get_processed_dimensions(img->w, img->h, &proc_w, &proc_h);

  int col_w = proc_w / 3;
  int bottom_y1 = (3 * proc_h) / 4;

  int orange_count[3] = {0, 0, 0};
  int green_count[3] = {0, 0, 0};
  int total_count[3] = {0, 0, 0};
  int dark_upper_count[3] = {0, 0, 0};
  int upper_total_count[3] = {0, 0, 0};

  for (int yr = 0; yr < proc_h; yr++) {
    for (int xr = 0; xr < proc_w; xr++) {
      int col = (xr < col_w) ? 0 : ((xr < (2 * col_w)) ? 1 : 2);

      int xs = 0;
      int ys = 0;
      map_processed_to_source(xr, yr, img->w, img->h, &xs, &ys);

      uint8_t yy = 0;
      uint8_t uu = 0;
      uint8_t vv = 0;
      if (!sample_yuv_uyvy(img, xs, ys, &yy, &uu, &vv)) {
        continue;
      }

      int h = 0;
      int s = 0;
      int val = 0;
      yuv_to_hsv(yy, uu, vv, &h, &s, &val);

      if (yr >= bottom_y1) {
        total_count[col]++;
        if (is_orange(h, s, val)) {
          orange_count[col]++;
        }

        // Down-weight floor-like green near the image bottom.
        if (yr < (bottom_y1 + ((proc_h - bottom_y1) * 3) / 4) && is_green(h, s, val)) {
          green_count[col]++;
        }
      }

      if (yr < (proc_h / 2)) {
        upper_total_count[col]++;
        if (yy < 70) {
          dark_upper_count[col]++;
        }
      }
    }
  }

  RegionMetrics metrics[3];
  for (int i = 0; i < 3; i++) {
    int total = std::max(1, total_count[i]);
    int upper_total = std::max(1, upper_total_count[i]);

    float orange_occ = (float)orange_count[i] / (float)total;
    float green_occ = (float)green_count[i] / (float)total;
    float window_occ = (float)dark_upper_count[i] / (float)upper_total;

    bool orange_blocked = orange_occ >= ORANGE_OCC_THRESHOLD;
    bool green_blocked = green_occ >= GREEN_OCC_THRESHOLD;
    bool window_blocked = window_occ <= WINDOW_LOW_THRESHOLD;

    int risk_score = 0;
    if (orange_blocked) {
      risk_score += 2;
    }
    if (green_blocked) {
      risk_score += 2;
    }
    if (window_blocked) {
      risk_score += 1;
    }

    metrics[i].orange_occ = orange_occ;
    metrics[i].green_occ = green_occ;
    metrics[i].window_occ = window_occ;
    metrics[i].orange_blocked = orange_blocked;
    metrics[i].green_blocked = green_blocked;
    metrics[i].window_blocked = window_blocked;
    metrics[i].blocked = orange_blocked || green_blocked;
    metrics[i].risk_score = risk_score;
  }

  *left_m = metrics[0];
  *center_m = metrics[1];
  *right_m = metrics[2];
}

static std::string decide_direction(const RegionMetrics &left_m,
                                    const RegionMetrics &center_m,
                                    const RegionMetrics &right_m)
{
  int left_risk = left_m.risk_score;
  int center_risk = center_m.risk_score;
  int right_risk = right_m.risk_score;

  VERBOSE_PRINT("Left Risk: %d\n", left_risk);
  VERBOSE_PRINT("Center Risk: %d\n", center_risk);
  VERBOSE_PRINT("Right Risk: %d\n", right_risk);

  if (left_risk == 0 && center_risk == 0 && right_risk == 0) {
    return "CENTER";
  }
  if (center_risk < 2) {
    return "CENTER";
  }
  if (!left_m.blocked && right_m.blocked) {
    return "LEFT";
  }
  if (!right_m.blocked && left_m.blocked) {
    return "RIGHT";
  }
  return (left_risk <= right_risk) ? "LEFT" : "RIGHT";
}

static void risk_to_binary_obstacles(const RegionMetrics &left_m,
                                     const RegionMetrics &center_m,
                                     const RegionMetrics &right_m,
                                     uint8_t *left_obs,
                                     uint8_t *center_obs,
                                     uint8_t *right_obs)
{
  *left_obs = (left_m.risk_score >= 2) ? 1 : 0;
  *center_obs = (center_m.risk_score >= 2) ? 1 : 0;
  *right_obs = (right_m.risk_score >= 2) ? 1 : 0;
}

static struct image_t *object_detector(struct image_t *img, uint8_t camera_id);
static struct image_t *object_detector(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
  auto start_time = std::chrono::high_resolution_clock::now();

  if (img == NULL || img->buf == NULL || img->w <= 0 || img->h <= 0) {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    VERBOSE_PRINT("object_detector execution time: %.3f ms\n", elapsed_us / 1000.0);
    return img;
  }

  RegionMetrics left_m, center_m, right_m;
  compute_column_metrics_from_yuv(img, &left_m, &center_m, &right_m);

  std::string decision = decide_direction(left_m, center_m, right_m);

  uint8_t left_obs = 0;
  uint8_t center_obs = 0;
  uint8_t right_obs = 0;
  risk_to_binary_obstacles(left_m, center_m, right_m, &left_obs, &center_obs, &right_obs);

  VERBOSE_PRINT("Decision: %s\n", decision.c_str());
  VERBOSE_PRINT("Binary obstacle message -> Left: %d Center: %d Right: %d\n",
                left_obs, center_obs, right_obs);

  pthread_mutex_lock(&detector_mutex);
  global_result.left = left_obs;
  global_result.middle = center_obs;
  global_result.right = right_obs;
  global_result.updated = true;
  pthread_mutex_unlock(&detector_mutex);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
  VERBOSE_PRINT("object_detector execution time: %.3f ms\n", elapsed_us / 1000.0);

  return img;
}

extern "C" {

void custom_detect_color_object_init(void)
{
  memset(&global_result, 0, sizeof(global_result));
  pthread_mutex_init(&detector_mutex, NULL);

#ifdef CUSTOM_DETECT_COLOR_OBJECT_CAMERA
  cv_add_to_device(&CUSTOM_DETECT_COLOR_OBJECT_CAMERA, object_detector, CUSTOM_DETECT_COLOR_OBJECT_FPS, 0);
#else
  PRINT("ERROR: CUSTOM_DETECT_COLOR_OBJECT_CAMERA is not defined in XML/airframe.\n");
#endif

  VERBOSE_PRINT("Initialized\n");
}

void custom_detect_color_object_periodic(void)
{
  struct custom_detection_result_t local_result;

  pthread_mutex_lock(&detector_mutex);
  memcpy(&local_result, &global_result, sizeof(local_result));
  global_result.updated = false;
  pthread_mutex_unlock(&detector_mutex);

  if (local_result.updated) {
    AbiSendMsgCUSTOM_DETECTION(CUSTOM_DETECT_COLOR_OBJECT_ID,
                               local_result.left,
                               local_result.middle,
                               local_result.right);

    VERBOSE_PRINT("Sent CUSTOM_DETECTION: left=%d middle=%d right=%d\n",
                  local_result.left, local_result.middle, local_result.right);
  }
}

}
