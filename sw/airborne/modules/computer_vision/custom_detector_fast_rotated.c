/*
 * Pure C Paparazzi detector derived from Python prototype:
 *  - orange YUV mask + vertical-band grouping
 *  - green YUV mask + edge-supported structured detection
 *  - 3-column risk evaluation
 *
 * Real path does not depend on OpenCV.
 * Optional debug drawing only modifies the incoming YUV image buffer.
 *
 * Rotation fix:
 *  - The Bebop front camera stream is rotated by 90 degrees.
 *  - We DO NOT rotate the image buffer physically.
 *  - We process the image in a logical coordinate frame using coordinate remapping.
 */

#include "modules/computer_vision/custom_detector_fast_rotated.h"
#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"
#include "std.h"

#include <pthread.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>

#ifndef CUSTOM_COLOR_RISK_DETECTOR_VERBOSE
#define CUSTOM_COLOR_RISK_DETECTOR_VERBOSE 1
#endif

#define PRINT(string,...) fprintf(stderr, "[custom_color_risk_detector->%s()] " string, __FUNCTION__, ##__VA_ARGS__)
#if CUSTOM_COLOR_RISK_DETECTOR_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

#ifndef CUSTOM_COLOR_RISK_DETECTOR_FPS
#define CUSTOM_COLOR_RISK_DETECTOR_FPS 0
#endif

#ifndef CUSTOM_COLOR_RISK_DETECTOR_DEBUG
#define CUSTOM_COLOR_RISK_DETECTOR_DEBUG false
#endif

#ifndef CUSTOM_COLOR_RISK_DETECTOR_CAMERA
#error "Define CUSTOM_COLOR_RISK_DETECTOR_CAMERA in the airframe/module settings"
#endif

#ifndef CUSTOM_DETECT_COLOR_OBJECT_ID
#define CUSTOM_DETECT_COLOR_OBJECT_ID 1
#endif

/* Rotation mode:
 * 0 = no rotation
 * 1 = logical 90 CW
 * 2 = logical 90 CCW
 *
 * Try 1 first. If left/right/up/down still look wrong, switch to 2.
 */
#ifndef CCRD_ROTATE_MODE
#define CCRD_ROTATE_MODE 2
#endif

/* ----------------------------
 * Thresholds from Python code
 * ---------------------------- */
#ifndef CCRD_ORANGE_Y_MIN
#define CCRD_ORANGE_Y_MIN 70
#endif
#ifndef CCRD_ORANGE_Y_MAX
#define CCRD_ORANGE_Y_MAX 170
#endif
#ifndef CCRD_ORANGE_U_MIN
#define CCRD_ORANGE_U_MIN 87
#endif
#ifndef CCRD_ORANGE_U_MAX
#define CCRD_ORANGE_U_MAX 121
#endif
#ifndef CCRD_ORANGE_V_MIN
#define CCRD_ORANGE_V_MIN 155
#endif
#ifndef CCRD_ORANGE_V_MAX
#define CCRD_ORANGE_V_MAX 210
#endif
#ifndef CCRD_ORANGE_VU_MIN
#define CCRD_ORANGE_VU_MIN 45
#endif

#ifndef CCRD_GREEN_Y_MIN
#define CCRD_GREEN_Y_MIN 45
#endif
#ifndef CCRD_GREEN_Y_MAX
#define CCRD_GREEN_Y_MAX 165
#endif
#ifndef CCRD_GREEN_U_MIN
#define CCRD_GREEN_U_MIN 50
#endif
#ifndef CCRD_GREEN_U_MAX
#define CCRD_GREEN_U_MAX 120
#endif
#ifndef CCRD_GREEN_V_MIN
#define CCRD_GREEN_V_MIN 105
#endif
#ifndef CCRD_GREEN_V_MAX
#define CCRD_GREEN_V_MAX 148
#endif
#ifndef CCRD_GREEN_VU_MIN
#define CCRD_GREEN_VU_MIN 20
#endif

#ifndef CCRD_ORANGE_ROI_Y_START_NUM
#define CCRD_ORANGE_ROI_Y_START_NUM 1
#endif
#ifndef CCRD_ORANGE_ROI_Y_START_DEN
#define CCRD_ORANGE_ROI_Y_START_DEN 10
#endif
#ifndef CCRD_ORANGE_ROI_Y_END_NUM
#define CCRD_ORANGE_ROI_Y_END_NUM 1
#endif
#ifndef CCRD_ORANGE_ROI_Y_END_DEN
#define CCRD_ORANGE_ROI_Y_END_DEN 1
#endif

#ifndef CCRD_ORANGE_SCAN_STEP_X
#define CCRD_ORANGE_SCAN_STEP_X 8
#endif
#ifndef CCRD_ORANGE_MIN_VERTICAL_RUN
#define CCRD_ORANGE_MIN_VERTICAL_RUN 10
#endif
#ifndef CCRD_ORANGE_MIN_HITS_IN_COLUMN
#define CCRD_ORANGE_MIN_HITS_IN_COLUMN 20
#endif
#ifndef CCRD_ORANGE_BAND_WIDTH
#define CCRD_ORANGE_BAND_WIDTH 8
#endif
#ifndef CCRD_ORANGE_MAX_GAP_BETWEEN_BANDS
#define CCRD_ORANGE_MAX_GAP_BETWEEN_BANDS 20
#endif
#ifndef CCRD_ORANGE_MIN_GROUP_WIDTH
#define CCRD_ORANGE_MIN_GROUP_WIDTH 6
#endif
#ifndef CCRD_ORANGE_MAX_GROUP_WIDTH
#define CCRD_ORANGE_MAX_GROUP_WIDTH 180
#endif
#ifndef CCRD_ORANGE_MIN_GROUP_HEIGHT
#define CCRD_ORANGE_MIN_GROUP_HEIGHT 35
#endif
#ifndef CCRD_ORANGE_MIN_GROUP_DENSITY_NUM
#define CCRD_ORANGE_MIN_GROUP_DENSITY_NUM 10
#endif
#ifndef CCRD_ORANGE_MIN_GROUP_DENSITY_DEN
#define CCRD_ORANGE_MIN_GROUP_DENSITY_DEN 100
#endif
#ifndef CCRD_ORANGE_MIN_ASPECT_NUM
#define CCRD_ORANGE_MIN_ASPECT_NUM 12
#endif
#ifndef CCRD_ORANGE_MIN_ASPECT_DEN
#define CCRD_ORANGE_MIN_ASPECT_DEN 10
#endif
#ifndef CCRD_ORANGE_VERTICAL_GAP_TOLERANCE
#define CCRD_ORANGE_VERTICAL_GAP_TOLERANCE 2
#endif

#ifndef CCRD_GRAD_THRESHOLD
#define CCRD_GRAD_THRESHOLD 18
#endif
#ifndef CCRD_GREEN_EDGE_EXPAND_PX
#define CCRD_GREEN_EDGE_EXPAND_PX 6
#endif
#ifndef CCRD_GREEN_MIN_SUPPORTED_PIXELS
#define CCRD_GREEN_MIN_SUPPORTED_PIXELS 30
#endif
#ifndef CCRD_GREEN_MIN_BOX_AREA
#define CCRD_GREEN_MIN_BOX_AREA 400
#endif
#ifndef CCRD_GREEN_MIN_HEIGHT
#define CCRD_GREEN_MIN_HEIGHT 20
#endif
#ifndef CCRD_GREEN_MIN_RAW_DENSITY_NUM
#define CCRD_GREEN_MIN_RAW_DENSITY_NUM 20
#endif
#ifndef CCRD_GREEN_MIN_RAW_DENSITY_DEN
#define CCRD_GREEN_MIN_RAW_DENSITY_DEN 100
#endif
#ifndef CCRD_GREEN_BBOX_EXPAND_X
#define CCRD_GREEN_BBOX_EXPAND_X 40
#endif
#ifndef CCRD_GREEN_BBOX_EXPAND_Y
#define CCRD_GREEN_BBOX_EXPAND_Y 10
#endif
#ifndef CCRD_GREEN_ROW_DENSITY_NUM
#define CCRD_GREEN_ROW_DENSITY_NUM 10
#endif
#ifndef CCRD_GREEN_ROW_DENSITY_DEN
#define CCRD_GREEN_ROW_DENSITY_DEN 100
#endif
#ifndef CCRD_GREEN_COL_DENSITY_NUM
#define CCRD_GREEN_COL_DENSITY_NUM 8
#endif
#ifndef CCRD_GREEN_COL_DENSITY_DEN
#define CCRD_GREEN_COL_DENSITY_DEN 100
#endif
#ifndef CCRD_GREEN_MIN_SUPPORTED_DENSITY_NUM
#define CCRD_GREEN_MIN_SUPPORTED_DENSITY_NUM 15
#endif
#ifndef CCRD_GREEN_MIN_SUPPORTED_DENSITY_DEN
#define CCRD_GREEN_MIN_SUPPORTED_DENSITY_DEN 1000
#endif
#ifndef CCRD_GREEN_MIN_ASPECT_NUM
#define CCRD_GREEN_MIN_ASPECT_NUM 45
#endif
#ifndef CCRD_GREEN_MIN_ASPECT_DEN
#define CCRD_GREEN_MIN_ASPECT_DEN 100
#endif
#ifndef CCRD_GREEN_MIN_SEED_GREEN_RATIO_NUM
#define CCRD_GREEN_MIN_SEED_GREEN_RATIO_NUM 8
#endif
#ifndef CCRD_GREEN_MIN_SEED_GREEN_RATIO_DEN
#define CCRD_GREEN_MIN_SEED_GREEN_RATIO_DEN 100
#endif

#ifndef CCRD_ORANGE_BOTTOM_ROWS_THRESHOLD_NUM
#define CCRD_ORANGE_BOTTOM_ROWS_THRESHOLD_NUM 15
#endif
#ifndef CCRD_ORANGE_BOTTOM_ROWS_THRESHOLD_DEN
#define CCRD_ORANGE_BOTTOM_ROWS_THRESHOLD_DEN 100
#endif
#ifndef CCRD_GREEN_MIDDLE_ROWS_THRESHOLD_NUM
#define CCRD_GREEN_MIDDLE_ROWS_THRESHOLD_NUM 15
#endif
#ifndef CCRD_GREEN_MIDDLE_ROWS_THRESHOLD_DEN
#define CCRD_GREEN_MIDDLE_ROWS_THRESHOLD_DEN 100
#endif

#ifndef CCRD_MAX_PIXELS
#define CCRD_MAX_PIXELS (640 * 480)
#endif
#ifndef CCRD_MAX_ORANGE_BANDS
#define CCRD_MAX_ORANGE_BANDS 512
#endif
#ifndef CCRD_MAX_ORANGE_OBJECTS
#define CCRD_MAX_ORANGE_OBJECTS 16
#endif
#ifndef CCRD_MAX_GREEN_COMPONENTS
#define CCRD_MAX_GREEN_COMPONENTS 384
#endif
#ifndef CCRD_MAX_GREEN_OBJECTS
#define CCRD_MAX_GREEN_OBJECTS 32
#endif

#ifndef CCRD_MAX_DIM
#define CCRD_MAX_DIM 640
#endif

struct orange_band_t {
  uint16_t x1, x2, y1, y2;
  uint16_t hits;
  uint16_t max_run;
};

struct bbox_t {
  uint16_t x1, y1, x2, y2;
};

struct component_t {
  uint16_t x1, y1, x2, y2;
  uint32_t area;
};

struct region_metrics_t {
  uint32_t orange_count;
  uint32_t green_count;
  uint32_t orange_total;
  uint32_t green_total;
  bool orange_blocked;
  bool green_blocked;
  bool blocked;
  uint8_t risk_score;
};

struct risk_result_t {
  uint8_t left;
  uint8_t middle;
  uint8_t right;
  struct region_metrics_t metrics[3];
  bool updated;
};

static pthread_mutex_t ccrd_mutex;
static struct risk_result_t g_result;

static uint8_t orange_mask[CCRD_MAX_PIXELS];
static uint8_t edge_mask[CCRD_MAX_PIXELS];
static uint8_t green_mask[CCRD_MAX_PIXELS];
static uint8_t green_supported[CCRD_MAX_PIXELS];
static uint8_t orange_valid[CCRD_MAX_PIXELS];
static uint8_t green_valid[CCRD_MAX_PIXELS];
static uint8_t visited[CCRD_MAX_PIXELS];

static uint16_t row_counts[CCRD_MAX_DIM];
static uint16_t col_counts[CCRD_MAX_DIM];
static uint32_t stack_buf[CCRD_MAX_PIXELS];

/* Helpers */

static inline uint32_t idx_of(uint16_t w, uint16_t x, uint16_t y)
{
  return ((uint32_t)y * (uint32_t)w) + (uint32_t)x;
}

static inline uint16_t ccrd_w(const struct image_t *img)
{
#if CCRD_ROTATE_MODE == 0
  return img->w;
#else
  return img->h;
#endif
}

static inline uint16_t ccrd_h(const struct image_t *img)
{
#if CCRD_ROTATE_MODE == 0
  return img->h;
#else
  return img->w;
#endif
}

static inline void ccrd_map_coords(const struct image_t *img,
                                   uint16_t lx, uint16_t ly,
                                   uint16_t *sx, uint16_t *sy)
{
#if CCRD_ROTATE_MODE == 0
  *sx = lx;
  *sy = ly;
#elif CCRD_ROTATE_MODE == 1
  /* logical 90 CW */
  *sx = ly;
  *sy = (uint16_t)(img->h - 1U - lx);
#elif CCRD_ROTATE_MODE == 2
  /* logical 90 CCW */
  *sx = (uint16_t)(img->w - 1U - ly);
  *sy = lx;
#else
  *sx = lx;
  *sy = ly;
#endif
}

static inline void yuv422_get_pixel(const struct image_t *img, uint16_t x, uint16_t y,
                                    uint8_t *yp, uint8_t *up, uint8_t *vp)
{
  uint8_t *buffer = img->buf;
  uint32_t base = (uint32_t)y * 2U * (uint32_t)img->w;

  if ((x & 1U) == 0U) {
    *up = buffer[base + 2U * x];
    *yp = buffer[base + 2U * x + 1U];
    *vp = buffer[base + 2U * x + 2U];
  } else {
    *up = buffer[base + 2U * x - 2U];
    *vp = buffer[base + 2U * x];
    *yp = buffer[base + 2U * x + 1U];
  }
}

static inline void yuv422_get_pixel_rot(const struct image_t *img, uint16_t x, uint16_t y,
                                        uint8_t *yp, uint8_t *up, uint8_t *vp)
{
  uint16_t sx, sy;
  ccrd_map_coords(img, x, y, &sx, &sy);
  yuv422_get_pixel(img, sx, sy, yp, up, vp);
}

static inline void yuv422_set_y(struct image_t *img, uint16_t x, uint16_t y, uint8_t value)
{
  uint8_t *buffer = img->buf;
  uint32_t base = (uint32_t)y * 2U * (uint32_t)img->w;
  buffer[base + 2U * x + 1U] = value;
}

static inline void yuv422_set_y_rot(struct image_t *img, uint16_t x, uint16_t y, uint8_t value)
{
  uint16_t sx, sy;
  ccrd_map_coords(img, x, y, &sx, &sy);
  yuv422_set_y(img, sx, sy, value);
}

static inline bool ratio_ge_u32(uint32_t count, uint32_t total, uint32_t num, uint32_t den)
{
  return (count * den) >= (total * num);
}

static void draw_rect_y(struct image_t *img, uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, uint8_t yval)
{
  if (!CUSTOM_COLOR_RISK_DETECTOR_DEBUG) {
    return;
  }
  if (x2 <= x1 || y2 <= y1) {
    return;
  }

  for (uint16_t x = x1; x < x2; x++) {
    yuv422_set_y_rot(img, x, y1, yval);
    yuv422_set_y_rot(img, x, (uint16_t)(y2 - 1U), yval);
  }
  for (uint16_t y = y1; y < y2; y++) {
    yuv422_set_y_rot(img, x1, y, yval);
    yuv422_set_y_rot(img, (uint16_t)(x2 - 1U), y, yval);
  }
}

static void draw_grid_debug(struct image_t *img)
{
  if (!CUSTOM_COLOR_RISK_DETECTOR_DEBUG) {
    return;
  }

  uint16_t w = ccrd_w(img);
  uint16_t h = ccrd_h(img);
  uint16_t x1 = (uint16_t)(w / 3U);
  uint16_t x2 = (uint16_t)((2U * w) / 3U);
  uint16_t y1 = (uint16_t)(h / 4U);
  uint16_t y2 = (uint16_t)(h / 2U);
  uint16_t y3 = (uint16_t)((3U * h) / 4U);

  for (uint16_t y = 0; y < h; y++) {
    yuv422_set_y_rot(img, x1, y, 235);
    yuv422_set_y_rot(img, x2, y, 235);
  }
  for (uint16_t x = 0; x < w; x++) {
    yuv422_set_y_rot(img, x, y1, 235);
    yuv422_set_y_rot(img, x, y2, 235);
    yuv422_set_y_rot(img, x, y3, 235);
  }
}

static void build_masks(const struct image_t *img)
{
  const uint16_t w = ccrd_w(img);
  const uint16_t h = ccrd_h(img);

  memset(orange_mask, 0, (size_t)w * h);
  memset(green_mask, 0, (size_t)w * h);
  memset(edge_mask, 0, (size_t)w * h);
  memset(orange_valid, 0, (size_t)w * h);
  memset(green_valid, 0, (size_t)w * h);

  for (uint16_t y = 0; y < h; y++) {

    if (w == 0U) {
      continue;
    }

    uint8_t y_prev = 0U, y_cur = 0U, y_next = 0U;
    uint8_t u_cur = 0U, v_cur = 0U;
    uint8_t u_tmp = 0U, v_tmp = 0U;

    yuv422_get_pixel_rot(img, 0U, y, &y_cur, &u_cur, &v_cur);

    {
      const uint32_t idx = idx_of(w, 0U, y);

      if (y_cur >= CCRD_ORANGE_Y_MIN && y_cur <= CCRD_ORANGE_Y_MAX &&
          u_cur >= CCRD_ORANGE_U_MIN && u_cur <= CCRD_ORANGE_U_MAX &&
          v_cur >= CCRD_ORANGE_V_MIN && v_cur <= CCRD_ORANGE_V_MAX &&
          ((int16_t)v_cur - (int16_t)u_cur) >= CCRD_ORANGE_VU_MIN) {
        orange_mask[idx] = 1U;
      }

      if (y_cur >= CCRD_GREEN_Y_MIN && y_cur <= CCRD_GREEN_Y_MAX &&
          u_cur >= CCRD_GREEN_U_MIN && u_cur <= CCRD_GREEN_U_MAX &&
          v_cur >= CCRD_GREEN_V_MIN && v_cur <= CCRD_GREEN_V_MAX &&
          ((int16_t)v_cur - (int16_t)u_cur) >= CCRD_GREEN_VU_MIN) {
        green_mask[idx] = 1U;
      }
    }

    if (w == 1U) {
      continue;
    }

    y_prev = y_cur;
    yuv422_get_pixel_rot(img, 1U, y, &y_cur, &u_cur, &v_cur);

    for (uint16_t x = 1U; x < w; x++) {
      const uint32_t idx = idx_of(w, x, y);

      if (y_cur >= CCRD_ORANGE_Y_MIN && y_cur <= CCRD_ORANGE_Y_MAX &&
          u_cur >= CCRD_ORANGE_U_MIN && u_cur <= CCRD_ORANGE_U_MAX &&
          v_cur >= CCRD_ORANGE_V_MIN && v_cur <= CCRD_ORANGE_V_MAX &&
          ((int16_t)v_cur - (int16_t)u_cur) >= CCRD_ORANGE_VU_MIN) {
        orange_mask[idx] = 1U;
      }

      if (y_cur >= CCRD_GREEN_Y_MIN && y_cur <= CCRD_GREEN_Y_MAX &&
          u_cur >= CCRD_GREEN_U_MIN && u_cur <= CCRD_GREEN_U_MAX &&
          v_cur >= CCRD_GREEN_V_MIN && v_cur <= CCRD_GREEN_V_MAX &&
          ((int16_t)v_cur - (int16_t)u_cur) >= CCRD_GREEN_VU_MIN) {
        green_mask[idx] = 1U;
      }

      if (x + 1U < w) {
        yuv422_get_pixel_rot(img, (uint16_t)(x + 1U), y, &y_next, &u_tmp, &v_tmp);

        int16_t grad = (int16_t)y_next - (int16_t)y_prev;
        if (grad < 0) {
          grad = -grad;
        }
        if (grad >= CCRD_GRAD_THRESHOLD) {
          edge_mask[idx] = 1U;
        }

        y_prev = y_cur;
        y_cur = y_next;
        u_cur = u_tmp;
        v_cur = v_tmp;
      }
    }
  }
}

static uint16_t scan_orange_vertical_bands(const struct image_t *img, struct orange_band_t *bands)
{
  const uint16_t w = ccrd_w(img);
  const uint16_t h = ccrd_h(img);
  const uint16_t roi_y1 = (uint16_t)((uint32_t)h * CCRD_ORANGE_ROI_Y_START_NUM / CCRD_ORANGE_ROI_Y_START_DEN);
  const uint16_t roi_y2 = (uint16_t)((uint32_t)h * CCRD_ORANGE_ROI_Y_END_NUM / CCRD_ORANGE_ROI_Y_END_DEN);
  uint16_t nbands = 0;

  for (uint16_t x = 0; x + CCRD_ORANGE_BAND_WIDTH <= w; x += CCRD_ORANGE_SCAN_STEP_X) {
    uint16_t hits = 0;
    uint8_t profile[1024];
    uint8_t fixed[1024];
    uint16_t roi_h = (uint16_t)(roi_y2 - roi_y1);

    if (roi_h > 1024U) {
      continue;
    }

    memset(profile, 0, roi_h);
    memset(fixed, 0, roi_h);

    for (uint16_t y = roi_y1; y < roi_y2; y++) {
      bool any = false;
      for (uint16_t dx = 0; dx < CCRD_ORANGE_BAND_WIDTH; dx++) {
        if (orange_mask[idx_of(w, (uint16_t)(x + dx), y)] != 0U) {
          hits++;
          any = true;
        }
      }
      profile[(uint16_t)(y - roi_y1)] = any ? 1U : 0U;
      fixed[(uint16_t)(y - roi_y1)] = profile[(uint16_t)(y - roi_y1)];
    }

    if (hits < CCRD_ORANGE_MIN_HITS_IN_COLUMN) {
      continue;
    }

    uint16_t zero_count = 0U;
    uint16_t start_zero = 0U;
    for (uint16_t i = 0; i < roi_h; i++) {
      if (profile[i] == 0U) {
        if (zero_count == 0U) {
          start_zero = i;
        }
        zero_count++;
      } else {
        if (zero_count > 0U && zero_count <= CCRD_ORANGE_VERTICAL_GAP_TOLERANCE) {
          for (uint16_t k = start_zero; k < i; k++) {
            fixed[k] = 1U;
          }
        }
        zero_count = 0U;
      }
    }
    if (zero_count > 0U && zero_count <= CCRD_ORANGE_VERTICAL_GAP_TOLERANCE) {
      for (uint16_t k = start_zero; k < roi_h; k++) {
        fixed[k] = 1U;
      }
    }

    uint16_t max_run = 0U, current_run = 0U, best_start = 0U, best_end = 0U, run_start = 0U;
    for (uint16_t i = 0; i < roi_h; i++) {
      if (fixed[i] != 0U) {
        if (current_run == 0U) {
          run_start = i;
        }
        current_run++;
        if (current_run > max_run) {
          max_run = current_run;
          best_start = run_start;
          best_end = i;
        }
      } else {
        current_run = 0U;
      }
    }

    if (max_run < CCRD_ORANGE_MIN_VERTICAL_RUN) {
      continue;
    }

    if (nbands < CCRD_MAX_ORANGE_BANDS) {
      bands[nbands].x1 = x;
      bands[nbands].x2 = (uint16_t)(x + CCRD_ORANGE_BAND_WIDTH);
      bands[nbands].y1 = (uint16_t)(roi_y1 + best_start);
      bands[nbands].y2 = (uint16_t)(roi_y1 + best_end + 1U);
      bands[nbands].hits = hits;
      bands[nbands].max_run = max_run;
      nbands++;
    }
  }

  return nbands;
}

static uint16_t group_orange_bands(const struct image_t *img,
                                   const struct orange_band_t *bands,
                                   uint16_t nbands,
                                   struct bbox_t *objects)
{
  const uint16_t w = ccrd_w(img);
  uint16_t nobj = 0;
  uint16_t start = 0;

  while (start < nbands) {
    uint16_t end = (uint16_t)(start + 1U);
    while (end < nbands) {
      const struct orange_band_t *prev = &bands[end - 1U];
      const struct orange_band_t *cur = &bands[end];
      int16_t gap_x = (int16_t)cur->x1 - (int16_t)prev->x2;
      uint16_t overlap_y1 = (prev->y1 > cur->y1) ? prev->y1 : cur->y1;
      uint16_t overlap_y2 = (prev->y2 < cur->y2) ? prev->y2 : cur->y2;
      uint16_t overlap_h = (overlap_y2 > overlap_y1) ? (uint16_t)(overlap_y2 - overlap_y1) : 0U;
      uint16_t prev_h = (uint16_t)(prev->y2 - prev->y1);
      uint16_t cur_h = (uint16_t)(cur->y2 - cur->y1);
      uint16_t min_h = (prev_h < cur_h) ? prev_h : cur_h;

      if (gap_x <= CCRD_ORANGE_MAX_GAP_BETWEEN_BANDS && (uint32_t)overlap_h * 100U >= (uint32_t)min_h * 35U) {
        end++;
      } else {
        break;
      }
    }

    uint16_t x1 = 65535U, y1 = 65535U, x2 = 0U, y2 = 0U;
    for (uint16_t i = start; i < end; i++) {
      if (bands[i].x1 < x1) { x1 = bands[i].x1; }
      if (bands[i].y1 < y1) { y1 = bands[i].y1; }
      if (bands[i].x2 > x2) { x2 = bands[i].x2; }
      if (bands[i].y2 > y2) { y2 = bands[i].y2; }
    }

    uint16_t bw = (uint16_t)(x2 - x1);
    uint16_t bh = (uint16_t)(y2 - y1);
    bool ok = true;
    if (bw < CCRD_ORANGE_MIN_GROUP_WIDTH || bw > CCRD_ORANGE_MAX_GROUP_WIDTH) { ok = false; }
    if (bh < CCRD_ORANGE_MIN_GROUP_HEIGHT) { ok = false; }
    if ((uint32_t)bh * CCRD_ORANGE_MIN_ASPECT_DEN < (uint32_t)bw * CCRD_ORANGE_MIN_ASPECT_NUM) { ok = false; }

    if (ok) {
      uint32_t cnt = 0U;
      uint32_t total = (uint32_t)bw * (uint32_t)bh;
      for (uint16_t y = y1; y < y2; y++) {
        for (uint16_t x = x1; x < x2; x++) {
          cnt += orange_mask[idx_of(w, x, y)] ? 1U : 0U;
        }
      }
      if (!ratio_ge_u32(cnt, total, CCRD_ORANGE_MIN_GROUP_DENSITY_NUM, CCRD_ORANGE_MIN_GROUP_DENSITY_DEN)) {
        ok = false;
      }
    }

    if (ok && nobj < CCRD_MAX_ORANGE_OBJECTS) {
      objects[nobj].x1 = x1;
      objects[nobj].y1 = y1;
      objects[nobj].x2 = x2;
      objects[nobj].y2 = y2;
      nobj++;

      for (uint16_t y = y1; y < y2; y++) {
        for (uint16_t x = x1; x < x2; x++) {
          orange_valid[idx_of(w, x, y)] = orange_mask[idx_of(w, x, y)];
        }
      }
    }

    start = end;
  }

  return nobj;
}

static uint16_t connected_components(const struct image_t *img, const uint8_t *mask, struct component_t *comps)
{
  const uint16_t w = ccrd_w(img);
  const uint16_t h = ccrd_h(img);
  const uint32_t npix = (uint32_t)w * (uint32_t)h;
  uint16_t ncomp = 0;

  memset(visited, 0, npix);

  for (uint16_t y = 0; y < h; y++) {
    for (uint16_t x = 0; x < w; x++) {
      const uint32_t start_idx = idx_of(w, x, y);
      if (mask[start_idx] == 0U || visited[start_idx] != 0U) {
        continue;
      }

      uint32_t sp = 0U;
      stack_buf[sp++] = start_idx;
      visited[start_idx] = 1U;
      uint32_t area = 0U;
      uint16_t x1 = x, x2 = x, y1 = y, y2 = y;

      while (sp > 0U) {
        uint32_t idx = stack_buf[--sp];
        uint16_t cx = (uint16_t)(idx % w);
        uint16_t cy = (uint16_t)(idx / w);
        area++;

        if (cx < x1) x1 = cx;
        if (cx > x2) x2 = cx;
        if (cy < y1) y1 = cy;
        if (cy > y2) y2 = cy;

        if (cx > 0U) {
          uint32_t n = idx - 1U;
          if (mask[n] != 0U && visited[n] == 0U) { visited[n] = 1U; stack_buf[sp++] = n; }
        }
        if (cx + 1U < w) {
          uint32_t n = idx + 1U;
          if (mask[n] != 0U && visited[n] == 0U) { visited[n] = 1U; stack_buf[sp++] = n; }
        }
        if (cy > 0U) {
          uint32_t n = idx - w;
          if (mask[n] != 0U && visited[n] == 0U) { visited[n] = 1U; stack_buf[sp++] = n; }
        }
        if (cy + 1U < h) {
          uint32_t n = idx + w;
          if (mask[n] != 0U && visited[n] == 0U) { visited[n] = 1U; stack_buf[sp++] = n; }
        }
      }

      if (ncomp < CCRD_MAX_GREEN_COMPONENTS) {
        comps[ncomp].x1 = x1;
        comps[ncomp].y1 = y1;
        comps[ncomp].x2 = (uint16_t)(x2 + 1U);
        comps[ncomp].y2 = (uint16_t)(y2 + 1U);
        comps[ncomp].area = area;
        ncomp++;
      }
    }
  }

  return ncomp;
}

static void build_green_supported(const struct image_t *img)
{
  const uint16_t w = ccrd_w(img);
  const uint16_t h = ccrd_h(img);
  memset(green_supported, 0, (size_t)w * h);

  for (uint16_t y = 0; y < h; y++) {
    for (uint16_t x = 0; x < w; x++) {
      uint8_t support = 0U;
      int16_t xmin = (int16_t)x - CCRD_GREEN_EDGE_EXPAND_PX;
      int16_t xmax = (int16_t)x + CCRD_GREEN_EDGE_EXPAND_PX;
      if (xmin < 0) xmin = 0;
      if (xmax >= (int16_t)w) xmax = (int16_t)w - 1;

      for (int16_t xx = xmin; xx <= xmax; xx++) {
        if (edge_mask[idx_of(w, (uint16_t)xx, y)] != 0U) {
          support = 1U;
          break;
        }
      }
      if (support && green_mask[idx_of(w, x, y)] != 0U) {
        green_supported[idx_of(w, x, y)] = 1U;
      }
    }
  }
}

static uint16_t detect_green_objects(const struct image_t *img, struct bbox_t *objects)
{
  const uint16_t w = ccrd_w(img);
  const uint16_t h = ccrd_h(img);
  struct component_t comps[CCRD_MAX_GREEN_COMPONENTS];
  uint16_t ncomp;
  uint16_t nobj = 0;

  build_green_supported(img);
  ncomp = connected_components(img, green_supported, comps);

  for (uint16_t i = 0; i < ncomp; i++) {
    if (comps[i].area < CCRD_GREEN_MIN_SUPPORTED_PIXELS) {
      continue;
    }

    uint16_t sx1 = comps[i].x1, sy1 = comps[i].y1, sx2 = comps[i].x2, sy2 = comps[i].y2;
    uint16_t x1e = (sx1 > CCRD_GREEN_BBOX_EXPAND_X) ? (uint16_t)(sx1 - CCRD_GREEN_BBOX_EXPAND_X) : 0U;
    uint16_t y1e = (sy1 > CCRD_GREEN_BBOX_EXPAND_Y) ? (uint16_t)(sy1 - CCRD_GREEN_BBOX_EXPAND_Y) : 0U;
    uint16_t x2e = (uint16_t)(((uint32_t)sx2 + CCRD_GREEN_BBOX_EXPAND_X < w) ? (sx2 + CCRD_GREEN_BBOX_EXPAND_X) : w);
    uint16_t y2e = (uint16_t)(((uint32_t)sy2 + CCRD_GREEN_BBOX_EXPAND_Y < h) ? (sy2 + CCRD_GREEN_BBOX_EXPAND_Y) : h);

    uint16_t roi_w = (uint16_t)(x2e - x1e);
    uint16_t roi_h = (uint16_t)(y2e - y1e);
    uint32_t roi_area = (uint32_t)roi_w * (uint32_t)roi_h;
    uint32_t raw_green_cnt = 0U;

    if (roi_h > CCRD_MAX_DIM || roi_w > CCRD_MAX_DIM) {
      continue;
    }

    memset(row_counts, 0, (size_t)roi_h * sizeof(uint16_t));
    memset(col_counts, 0, (size_t)roi_w * sizeof(uint16_t));

    for (uint16_t y = y1e; y < y2e; y++) {
      for (uint16_t x = x1e; x < x2e; x++) {
        if (green_mask[idx_of(w, x, y)] != 0U) {
          raw_green_cnt++;
          row_counts[y - y1e]++;
          col_counts[x - x1e]++;
        }
      }
    }

    if (!ratio_ge_u32(raw_green_cnt, roi_area,
                      CCRD_GREEN_MIN_SEED_GREEN_RATIO_NUM,
                      CCRD_GREEN_MIN_SEED_GREEN_RATIO_DEN)) {
      continue;
    }

    int16_t first_row = -1, last_row = -1, first_col = -1, last_col = -1;
    for (uint16_t r = 0; r < roi_h; r++) {
      if ((uint32_t)row_counts[r] * CCRD_GREEN_ROW_DENSITY_DEN >= (uint32_t)roi_w * CCRD_GREEN_ROW_DENSITY_NUM) {
        if (first_row < 0) first_row = (int16_t)r;
        last_row = (int16_t)r;
      }
    }
    for (uint16_t c = 0; c < roi_w; c++) {
      if ((uint32_t)col_counts[c] * CCRD_GREEN_COL_DENSITY_DEN >= (uint32_t)roi_h * CCRD_GREEN_COL_DENSITY_NUM) {
        if (first_col < 0) first_col = (int16_t)c;
        last_col = (int16_t)c;
      }
    }

    if (first_row < 0 || first_col < 0) {
      continue;
    }

    uint16_t gx1 = (uint16_t)(x1e + first_col);
    uint16_t gx2 = (uint16_t)(x1e + last_col + 1);
    uint16_t gy1 = (uint16_t)(y1e + first_row);
    uint16_t gy2 = (uint16_t)(y1e + last_row + 1);
    uint16_t bw = (uint16_t)(gx2 - gx1);
    uint16_t bh = (uint16_t)(gy2 - gy1);
    uint32_t box_area = (uint32_t)bw * (uint32_t)bh;

    if (box_area < CCRD_GREEN_MIN_BOX_AREA) {
      continue;
    }
    if (bh < CCRD_GREEN_MIN_HEIGHT) {
      continue;
    }
    if ((uint32_t)bh * CCRD_GREEN_MIN_ASPECT_DEN < (uint32_t)bw * CCRD_GREEN_MIN_ASPECT_NUM) {
      continue;
    }

    uint32_t raw_cnt = 0U;
    uint32_t supp_cnt = 0U;
    for (uint16_t y = gy1; y < gy2; y++) {
      for (uint16_t x = gx1; x < gx2; x++) {
        raw_cnt += green_mask[idx_of(w, x, y)] ? 1U : 0U;
        supp_cnt += green_supported[idx_of(w, x, y)] ? 1U : 0U;
      }
    }

    if (!ratio_ge_u32(raw_cnt, box_area, CCRD_GREEN_MIN_RAW_DENSITY_NUM, CCRD_GREEN_MIN_RAW_DENSITY_DEN)) {
      continue;
    }
    if (!ratio_ge_u32(supp_cnt, box_area,
                      CCRD_GREEN_MIN_SUPPORTED_DENSITY_NUM,
                      CCRD_GREEN_MIN_SUPPORTED_DENSITY_DEN)) {
      continue;
    }

    if (nobj < CCRD_MAX_GREEN_OBJECTS) {
      objects[nobj].x1 = gx1;
      objects[nobj].y1 = gy1;
      objects[nobj].x2 = gx2;
      objects[nobj].y2 = gy2;
      nobj++;
    }

    for (uint16_t y = gy1; y < gy2; y++) {
      for (uint16_t x = gx1; x < gx2; x++) {
        green_valid[idx_of(w, x, y)] = green_supported[idx_of(w, x, y)];
      }
    }
  }

  return nobj;
}

static void compute_risk_map(const struct image_t *img, struct risk_result_t *result)
{
  const uint16_t w = ccrd_w(img);
  const uint16_t h = ccrd_h(img);
  const uint16_t col_edges[4] = {0U, (uint16_t)(w / 3U), (uint16_t)((2U * w) / 3U), w};
  const uint16_t row_edges[5] = {0U, (uint16_t)(h / 4U), (uint16_t)(h / 2U), (uint16_t)((3U * h) / 4U), h};

  const uint16_t orange_y1 = row_edges[3];
  const uint16_t orange_y2 = row_edges[4];
  const uint16_t green_y1 = row_edges[1];
  const uint16_t green_y2 = row_edges[3];

  result->left = 0U;
  result->middle = 0U;
  result->right = 0U;

  for (uint8_t i = 0U; i < 3U; i++) {
    uint16_t x1 = col_edges[i];
    uint16_t x2 = col_edges[i + 1U];
    uint32_t orange_count = 0U;
    uint32_t green_count = 0U;
    uint32_t orange_total = (uint32_t)(x2 - x1) * (uint32_t)(orange_y2 - orange_y1);
    uint32_t green_total = (uint32_t)(x2 - x1) * (uint32_t)(green_y2 - green_y1);

    for (uint16_t y = orange_y1; y < orange_y2; y++) {
      for (uint16_t x = x1; x < x2; x++) {
        orange_count += orange_valid[idx_of(w, x, y)] ? 1U : 0U;
      }
    }
    for (uint16_t y = green_y1; y < green_y2; y++) {
      for (uint16_t x = x1; x < x2; x++) {
        green_count += green_valid[idx_of(w, x, y)] ? 1U : 0U;
      }
    }

    result->metrics[i].orange_count = orange_count;
    result->metrics[i].green_count = green_count;
    result->metrics[i].orange_total = orange_total;
    result->metrics[i].green_total = green_total;
    result->metrics[i].orange_blocked = ratio_ge_u32(orange_count, orange_total,
                                                     CCRD_ORANGE_BOTTOM_ROWS_THRESHOLD_NUM,
                                                     CCRD_ORANGE_BOTTOM_ROWS_THRESHOLD_DEN);
    result->metrics[i].green_blocked = ratio_ge_u32(green_count, green_total,
                                                    CCRD_GREEN_MIDDLE_ROWS_THRESHOLD_NUM,
                                                    CCRD_GREEN_MIDDLE_ROWS_THRESHOLD_DEN);
    result->metrics[i].blocked = result->metrics[i].orange_blocked || result->metrics[i].green_blocked;
    result->metrics[i].risk_score = (result->metrics[i].orange_blocked ? 2U : 0U) +
                                    (result->metrics[i].green_blocked ? 2U : 0U);

    if (CUSTOM_COLOR_RISK_DETECTOR_DEBUG) {
      draw_rect_y((struct image_t *)img, x1, orange_y1, x2, orange_y2,
                  result->metrics[i].orange_blocked ? 200 : 120);
      draw_rect_y((struct image_t *)img, x1, green_y1, x2, green_y2,
                  result->metrics[i].green_blocked ? 255 : 80);
      draw_rect_y((struct image_t *)img, x1, 0U, x2, h,
                  result->metrics[i].blocked ? 255 : 180);
    }
  }

  result->left = result->metrics[0].blocked ? 1U : 0U;
  result->middle = result->metrics[1].blocked ? 1U : 0U;
  result->right = result->metrics[2].blocked ? 1U : 0U;
}

static struct image_t *custom_color_risk_detector_func(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
  struct orange_band_t bands[CCRD_MAX_ORANGE_BANDS];
  struct bbox_t orange_objects[CCRD_MAX_ORANGE_OBJECTS];
  struct bbox_t green_objects[CCRD_MAX_GREEN_OBJECTS];
  struct risk_result_t local_result;

  const uint32_t npix = (uint32_t)img->w * (uint32_t)img->h;
  if (npix > CCRD_MAX_PIXELS || img->type != IMAGE_YUV422) {
    return img;
  }

  memset(&local_result, 0, sizeof(local_result));

  build_masks(img);
  uint16_t nbands = scan_orange_vertical_bands(img, bands);
  uint16_t norange = group_orange_bands(img, bands, nbands, orange_objects);
  uint16_t ngreen = detect_green_objects(img, green_objects);
  compute_risk_map(img, &local_result);
  local_result.updated = true;

  if (CUSTOM_COLOR_RISK_DETECTOR_DEBUG) {
    draw_grid_debug(img);
    for (uint16_t i = 0; i < norange; i++) {
      draw_rect_y(img, orange_objects[i].x1, orange_objects[i].y1, orange_objects[i].x2, orange_objects[i].y2, 220);
    }
    for (uint16_t i = 0; i < ngreen; i++) {
      draw_rect_y(img, green_objects[i].x1, green_objects[i].y1, green_objects[i].x2, green_objects[i].y2, 150);
    }
  }

  VERBOSE_PRINT("orange objects=%u green objects=%u risk=[%u,%u,%u]\n",
                norange, ngreen, local_result.left, local_result.middle, local_result.right);

  pthread_mutex_lock(&ccrd_mutex);
  memcpy(&g_result, &local_result, sizeof(local_result));
  pthread_mutex_unlock(&ccrd_mutex);

  return img;
}

void custom_color_risk_detector_init(void)
{
  memset(&g_result, 0, sizeof(g_result));
  pthread_mutex_init(&ccrd_mutex, NULL);
  cv_add_to_device(&CUSTOM_COLOR_RISK_DETECTOR_CAMERA,
                   custom_color_risk_detector_func,
                   CUSTOM_COLOR_RISK_DETECTOR_FPS,
                   0);
}

void custom_color_risk_detector_periodic(void)
{
  struct risk_result_t local_result;
  pthread_mutex_lock(&ccrd_mutex);
  memcpy(&local_result, &g_result, sizeof(local_result));
  g_result.updated = false;
  pthread_mutex_unlock(&ccrd_mutex);

  if (local_result.updated) {
    AbiSendMsgCUSTOM_DETECTION(CUSTOM_DETECT_COLOR_OBJECT_ID,
                               local_result.left,
                               local_result.middle,
                               local_result.right);
  }
}