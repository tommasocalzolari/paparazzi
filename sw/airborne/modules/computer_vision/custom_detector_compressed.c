/*
 * Pure C Paparazzi detector derived from standalone_detector_viewer.cpp
 * but rewritten for Paparazzi:
 *   - no OpenCV
 *   - optional UYVY/YUV422 downsample
 *   - orange YUV mask + vertical-band grouping
 *   - green YUV mask + edge-supported structured detection
 *   - 3-column risk evaluation
 *   - debug drawing directly in Y channel
 *   - timing logged to file
 */

#include "modules/computer_vision/custom_detector_fast_timer.h"
#include "modules/core/abi.h"
#include "std.h"

#include <pthread.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include <sys/time.h>

#ifndef CUSTOM_COLOR_RISK_DETECTOR_VERBOSE
#define CUSTOM_COLOR_RISK_DETECTOR_VERBOSE 1
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

#ifndef CCRD_ROTATE_MODE
#define CCRD_ROTATE_MODE 1
#endif

#ifndef CCRD_ENABLE_DOWNSAMPLE
#define CCRD_ENABLE_DOWNSAMPLE 1
#endif

#ifndef CCRD_DOWNSAMPLE
#define CCRD_DOWNSAMPLE 2
#endif

/* ----------------------------
 * Thresholds
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

/* downsample=2 defaults */
#ifndef CCRD_ORANGE_SCAN_STEP_X
#define CCRD_ORANGE_SCAN_STEP_X 4
#endif
#ifndef CCRD_ORANGE_MIN_VERTICAL_RUN
#define CCRD_ORANGE_MIN_VERTICAL_RUN 5
#endif
#ifndef CCRD_ORANGE_MIN_HITS_IN_COLUMN
#define CCRD_ORANGE_MIN_HITS_IN_COLUMN 8
#endif
#ifndef CCRD_ORANGE_BAND_WIDTH
#define CCRD_ORANGE_BAND_WIDTH 4
#endif
#ifndef CCRD_ORANGE_MAX_GAP_BETWEEN_BANDS
#define CCRD_ORANGE_MAX_GAP_BETWEEN_BANDS 10
#endif
#ifndef CCRD_ORANGE_MIN_GROUP_WIDTH
#define CCRD_ORANGE_MIN_GROUP_WIDTH 3
#endif
#ifndef CCRD_ORANGE_MAX_GROUP_WIDTH
#define CCRD_ORANGE_MAX_GROUP_WIDTH 90
#endif
#ifndef CCRD_ORANGE_MIN_GROUP_HEIGHT
#define CCRD_ORANGE_MIN_GROUP_HEIGHT 18
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
#define CCRD_ORANGE_VERTICAL_GAP_TOLERANCE 1
#endif

#ifndef CCRD_GRAD_THRESHOLD
#define CCRD_GRAD_THRESHOLD 12
#endif
#ifndef CCRD_GREEN_EDGE_EXPAND_PX
#define CCRD_GREEN_EDGE_EXPAND_PX 3
#endif
#ifndef CCRD_GREEN_MIN_SUPPORTED_PIXELS
#define CCRD_GREEN_MIN_SUPPORTED_PIXELS 10
#endif
#ifndef CCRD_GREEN_MIN_BOX_AREA
#define CCRD_GREEN_MIN_BOX_AREA 100
#endif
#ifndef CCRD_GREEN_MIN_HEIGHT
#define CCRD_GREEN_MIN_HEIGHT 10
#endif
#ifndef CCRD_GREEN_MIN_RAW_DENSITY_NUM
#define CCRD_GREEN_MIN_RAW_DENSITY_NUM 20
#endif
#ifndef CCRD_GREEN_MIN_RAW_DENSITY_DEN
#define CCRD_GREEN_MIN_RAW_DENSITY_DEN 100
#endif
#ifndef CCRD_GREEN_BBOX_EXPAND_X
#define CCRD_GREEN_BBOX_EXPAND_X 20
#endif
#ifndef CCRD_GREEN_BBOX_EXPAND_Y
#define CCRD_GREEN_BBOX_EXPAND_Y 5
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
#define CCRD_GREEN_MIDDLE_ROWS_THRESHOLD_NUM 10
#endif
#ifndef CCRD_GREEN_MIDDLE_ROWS_THRESHOLD_DEN
#define CCRD_GREEN_MIDDLE_ROWS_THRESHOLD_DEN 100
#endif

#ifndef CCRD_MAX_PIXELS
#define CCRD_MAX_PIXELS (640U * 480U)
#endif
#ifndef CCRD_MAX_DOWNSAMPLED_BYTES
#define CCRD_MAX_DOWNSAMPLED_BYTES (CCRD_MAX_PIXELS * 2U)
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

#ifndef CCRD_LOG_EVERY_N_FRAMES
#define CCRD_LOG_EVERY_N_FRAMES 20U
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
static uint8_t y_logical[CCRD_MAX_PIXELS];

static uint16_t row_counts[CCRD_MAX_DIM];
static uint16_t col_counts[CCRD_MAX_DIM];
static uint32_t stack_buf[CCRD_MAX_PIXELS];

static uint8_t downsample_buf[CCRD_MAX_DOWNSAMPLED_BYTES];
static struct image_t downsampled_img;

static FILE *ccrd_log_file = NULL;
static uint32_t ccrd_frame_counter = 0U;

static void ccrd_log(const char *fmt, ...);

#define PRINT(string,...) ccrd_log("[custom_color_risk_detector->%s()] " string, __FUNCTION__, ##__VA_ARGS__)
#if CUSTOM_COLOR_RISK_DETECTOR_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

static uint64_t now_us(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
}

static void ccrd_open_log_file(void)
{
  if (ccrd_log_file == NULL) {
    ccrd_log_file = fopen("/data/video/custom_detector_log.txt", "a");
    if (ccrd_log_file == NULL) {
      ccrd_log_file = fopen("/tmp/custom_detector_log.txt", "a");
    }
  }
}

static void ccrd_log(const char *fmt, ...)
{
  ccrd_open_log_file();
  if (ccrd_log_file == NULL) {
    return;
  }

  va_list args;
  va_start(args, fmt);
  vfprintf(ccrd_log_file, fmt, args);
  va_end(args);
  fflush(ccrd_log_file);
}

static inline uint32_t idx_of(uint16_t w, uint16_t x, uint16_t y)
{
  return ((uint32_t)y * (uint32_t)w) + (uint32_t)x;
}

static inline bool ratio_ge_u32(uint32_t count, uint32_t total, uint32_t num, uint32_t den)
{
  return (count * den) >= (total * num);
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
  *sx = ly;
  *sy = (uint16_t)(img->h - 1U - lx);
#else
  *sx = (uint16_t)(img->w - 1U - ly);
  *sy = lx;
#endif
}

static inline void yuv422_get_pixel(const struct image_t *img, uint16_t x, uint16_t y,
                                    uint8_t *yp, uint8_t *up, uint8_t *vp)
{
  uint8_t *buffer = (uint8_t *)img->buf;
  uint32_t base = (uint32_t)y * 2U * (uint32_t)img->w;

  if ((x & 1U) == 0U) {
    *up = buffer[base + 2U * x + 0U];
    *yp = buffer[base + 2U * x + 1U];
    *vp = buffer[base + 2U * x + 2U];
  } else {
    *up = buffer[base + 2U * x - 2U];
    *vp = buffer[base + 2U * x + 0U];
    *yp = buffer[base + 2U * x + 1U];
  }
}

static inline void yuv422_set_y(struct image_t *img, uint16_t x, uint16_t y, uint8_t value)
{
  uint8_t *buffer = (uint8_t *)img->buf;
  uint32_t base = (uint32_t)y * 2U * (uint32_t)img->w;
  buffer[base + 2U * x + 1U] = value;
}

static inline void yuv422_set_y_rot(struct image_t *img, uint16_t x, uint16_t y, uint8_t value)
{
  uint16_t sx, sy;
  ccrd_map_coords(img, x, y, &sx, &sy);
  yuv422_set_y(img, sx, sy, value);
}

static void draw_rect_y(struct image_t *img, uint16_t x1, uint16_t y1,
                        uint16_t x2, uint16_t y2, uint8_t yval)
{
  uint16_t x, y;
  if (!CUSTOM_COLOR_RISK_DETECTOR_DEBUG) {
    return;
  }
  if (x2 <= x1 || y2 <= y1) {
    return;
  }

  for (x = x1; x < x2; x++) {
    yuv422_set_y_rot(img, x, y1, yval);
    yuv422_set_y_rot(img, x, (uint16_t)(y2 - 1U), yval);
  }
  for (y = y1; y < y2; y++) {
    yuv422_set_y_rot(img, x1, y, yval);
    yuv422_set_y_rot(img, (uint16_t)(x2 - 1U), y, yval);
  }
}

static void draw_grid_debug(struct image_t *img)
{
  uint16_t w, h, x1, x2, y1, y2, y3, x, y;
  if (!CUSTOM_COLOR_RISK_DETECTOR_DEBUG) {
    return;
  }

  w = ccrd_w(img);
  h = ccrd_h(img);
  x1 = (uint16_t)(w / 3U);
  x2 = (uint16_t)((2U * w) / 3U);
  y1 = (uint16_t)(h / 4U);
  y2 = (uint16_t)(h / 2U);
  y3 = (uint16_t)((3U * h) / 4U);

  for (y = 0U; y < h; y++) {
    yuv422_set_y_rot(img, x1, y, 235U);
    yuv422_set_y_rot(img, x2, y, 235U);
  }
  for (x = 0U; x < w; x++) {
    yuv422_set_y_rot(img, x, y1, 235U);
    yuv422_set_y_rot(img, x, y2, 235U);
    yuv422_set_y_rot(img, x, y3, 235U);
  }
}

static void build_masks(const struct image_t *img)
{
  const uint16_t lw = ccrd_w(img);
  const uint16_t lh = ccrd_h(img);
  const uint16_t sw = img->w;
  const uint16_t sh = img->h;
  uint8_t *buf = (uint8_t *)img->buf;
  uint16_t sy, sx_pair;

  memset(orange_mask, 0, (size_t)lw * lh);
  memset(green_mask, 0, (size_t)lw * lh);
  memset(edge_mask, 0, (size_t)lw * lh);
  memset(orange_valid, 0, (size_t)lw * lh);
  memset(green_valid, 0, (size_t)lw * lh);
  memset(y_logical, 0, (size_t)lw * lh);

  for (sy = 0U; sy < sh; sy++) {
    uint8_t *row = &buf[(uint32_t)sy * 2U * (uint32_t)sw];

    for (sx_pair = 0U; sx_pair + 1U < sw; sx_pair += 2U) {
      uint8_t u  = row[2U * sx_pair + 0U];
      uint8_t y0 = row[2U * sx_pair + 1U];
      uint8_t v  = row[2U * sx_pair + 2U];
      uint8_t y1 = row[2U * sx_pair + 3U];
      uint8_t k;

      for (k = 0U; k < 2U; k++) {
        uint16_t sx = (uint16_t)(sx_pair + k);
        uint8_t yy = (k == 0U) ? y0 : y1;
        uint16_t lx, ly;
        int16_t vu;
        uint32_t idx;

#if CCRD_ROTATE_MODE == 0
        lx = sx; ly = sy;
#elif CCRD_ROTATE_MODE == 1
        lx = (uint16_t)(img->h - 1U - sy); ly = sx;
#else
        lx = sy; ly = (uint16_t)(img->w - 1U - sx);
#endif

        if (lx >= lw || ly >= lh) {
          continue;
        }

        idx = idx_of(lw, lx, ly);
        vu = (int16_t)v - (int16_t)u;
        y_logical[idx] = yy;

        if (yy >= CCRD_ORANGE_Y_MIN && yy <= CCRD_ORANGE_Y_MAX &&
            u  >= CCRD_ORANGE_U_MIN && u  <= CCRD_ORANGE_U_MAX &&
            v  >= CCRD_ORANGE_V_MIN && v  <= CCRD_ORANGE_V_MAX &&
            vu >= CCRD_ORANGE_VU_MIN) {
          orange_mask[idx] = 1U;
        }

        if (yy >= CCRD_GREEN_Y_MIN && yy <= CCRD_GREEN_Y_MAX &&
            u  >= CCRD_GREEN_U_MIN && u  <= CCRD_GREEN_U_MAX &&
            v  >= CCRD_GREEN_V_MIN && v  <= CCRD_GREEN_V_MAX &&
            vu >= CCRD_GREEN_VU_MIN) {
          green_mask[idx] = 1U;
        }
      }
    }
  }

  if (lw >= 3U) {
    uint16_t y, x;
    for (y = 0U; y < lh; y++) {
      for (x = 1U; x + 1U < lw; x++) {
        uint8_t y_prev = y_logical[idx_of(lw, (uint16_t)(x - 1U), y)];
        uint8_t y_next = y_logical[idx_of(lw, (uint16_t)(x + 1U), y)];
        int16_t grad = (int16_t)y_next - (int16_t)y_prev;
        if (grad < 0) {
          grad = -grad;
        }
        if (grad >= CCRD_GRAD_THRESHOLD) {
          edge_mask[idx_of(lw, x, y)] = 1U;
        }
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
  uint16_t nbands = 0U;
  uint16_t x;

  for (x = 0U; x + CCRD_ORANGE_BAND_WIDTH <= w; x += CCRD_ORANGE_SCAN_STEP_X) {
    uint16_t hits = 0U;
    uint8_t profile[1024];
    uint8_t fixed[1024];
    uint16_t roi_h = (uint16_t)(roi_y2 - roi_y1);
    uint16_t y;

    if (roi_h > 1024U) {
      continue;
    }

    memset(profile, 0, roi_h);
    memset(fixed, 0, roi_h);

    for (y = roi_y1; y < roi_y2; y++) {
      bool any = false;
      uint16_t dx;

      for (dx = 0U; dx < CCRD_ORANGE_BAND_WIDTH; dx++) {
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

    {
      uint16_t zero_count = 0U;
      uint16_t start_zero = 0U;
      uint16_t i;

      for (i = 0U; i < roi_h; i++) {
        if (profile[i] == 0U) {
          if (zero_count == 0U) {
            start_zero = i;
          }
          zero_count++;
        } else {
          if (zero_count > 0U && zero_count <= CCRD_ORANGE_VERTICAL_GAP_TOLERANCE) {
            uint16_t k;
            for (k = start_zero; k < i; k++) {
              fixed[k] = 1U;
            }
          }
          zero_count = 0U;
        }
      }

      if (zero_count > 0U && zero_count <= CCRD_ORANGE_VERTICAL_GAP_TOLERANCE) {
        uint16_t k;
        for (k = start_zero; k < roi_h; k++) {
          fixed[k] = 1U;
        }
      }
    }

    {
      uint16_t max_run = 0U, current_run = 0U, best_start = 0U, best_end = 0U, run_start = 0U;
      uint16_t i;

      for (i = 0U; i < roi_h; i++) {
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
  }

  return nbands;
}

static uint16_t group_orange_bands(const struct image_t *img,
                                   const struct orange_band_t *bands,
                                   uint16_t nbands,
                                   struct bbox_t *objects)
{
  const uint16_t w = ccrd_w(img);
  uint16_t nobj = 0U;
  uint16_t start = 0U;

  while (start < nbands) {
    uint16_t end = (uint16_t)(start + 1U);

    while (end < nbands) {
      const struct orange_band_t *prev = &bands[end - 1U];
      const struct orange_band_t *cur  = &bands[end];
      int16_t gap_x = (int16_t)cur->x1 - (int16_t)prev->x2;
      uint16_t overlap_y1 = (prev->y1 > cur->y1) ? prev->y1 : cur->y1;
      uint16_t overlap_y2 = (prev->y2 < cur->y2) ? prev->y2 : cur->y2;
      uint16_t overlap_h = (overlap_y2 > overlap_y1) ? (uint16_t)(overlap_y2 - overlap_y1) : 0U;
      uint16_t prev_h = (uint16_t)(prev->y2 - prev->y1);
      uint16_t cur_h  = (uint16_t)(cur->y2 - cur->y1);
      uint16_t min_h = (prev_h < cur_h) ? prev_h : cur_h;

      if (gap_x <= CCRD_ORANGE_MAX_GAP_BETWEEN_BANDS &&
          (uint32_t)overlap_h * 100U >= (uint32_t)min_h * 35U) {
        end++;
      } else {
        break;
      }
    }

    {
      uint16_t x1 = 65535U, y1 = 65535U, x2 = 0U, y2 = 0U;
      uint16_t i;
      bool ok = true;
      uint16_t bw, bh;

      for (i = start; i < end; i++) {
        if (bands[i].x1 < x1) { x1 = bands[i].x1; }
        if (bands[i].y1 < y1) { y1 = bands[i].y1; }
        if (bands[i].x2 > x2) { x2 = bands[i].x2; }
        if (bands[i].y2 > y2) { y2 = bands[i].y2; }
      }

      bw = (uint16_t)(x2 - x1);
      bh = (uint16_t)(y2 - y1);

      if (bw < CCRD_ORANGE_MIN_GROUP_WIDTH || bw > CCRD_ORANGE_MAX_GROUP_WIDTH) ok = false;
      if (bh < CCRD_ORANGE_MIN_GROUP_HEIGHT) ok = false;
      if ((uint32_t)bh * CCRD_ORANGE_MIN_ASPECT_DEN < (uint32_t)bw * CCRD_ORANGE_MIN_ASPECT_NUM) ok = false;

      if (ok) {
        uint32_t cnt = 0U;
        uint32_t total = (uint32_t)bw * (uint32_t)bh;
        uint16_t x, y;

        for (y = y1; y < y2; y++) {
          for (x = x1; x < x2; x++) {
            cnt += orange_mask[idx_of(w, x, y)] ? 1U : 0U;
          }
        }

        if (!ratio_ge_u32(cnt, total,
                          CCRD_ORANGE_MIN_GROUP_DENSITY_NUM,
                          CCRD_ORANGE_MIN_GROUP_DENSITY_DEN)) {
          ok = false;
        }
      }

      if (ok && nobj < CCRD_MAX_ORANGE_OBJECTS) {
        uint16_t x, y;
        objects[nobj].x1 = x1;
        objects[nobj].y1 = y1;
        objects[nobj].x2 = x2;
        objects[nobj].y2 = y2;
        nobj++;

        for (y = y1; y < y2; y++) {
          for (x = x1; x < x2; x++) {
            orange_valid[idx_of(w, x, y)] = orange_mask[idx_of(w, x, y)];
          }
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
  uint16_t ncomp = 0U;
  uint16_t x, y;

  memset(visited, 0, (size_t)w * h);

  for (y = 0U; y < h; y++) {
    for (x = 0U; x < w; x++) {
      uint32_t start_idx = idx_of(w, x, y);
      uint32_t sp;
      uint32_t area;
      uint16_t x1, x2, y1, y2;

      if (mask[start_idx] == 0U || visited[start_idx] != 0U) {
        continue;
      }

      sp = 0U;
      stack_buf[sp++] = start_idx;
      visited[start_idx] = 1U;
      area = 0U;
      x1 = x; x2 = x; y1 = y; y2 = y;

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
          if (mask[n] != 0U && visited[n] == 0U) {
            visited[n] = 1U; stack_buf[sp++] = n;
          }
        }
        if (cx + 1U < w) {
          uint32_t n = idx + 1U;
          if (mask[n] != 0U && visited[n] == 0U) {
            visited[n] = 1U; stack_buf[sp++] = n;
          }
        }
        if (cy > 0U) {
          uint32_t n = idx - w;
          if (mask[n] != 0U && visited[n] == 0U) {
            visited[n] = 1U; stack_buf[sp++] = n;
          }
        }
        if (cy + 1U < h) {
          uint32_t n = idx + w;
          if (mask[n] != 0U && visited[n] == 0U) {
            visited[n] = 1U; stack_buf[sp++] = n;
          }
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
  uint16_t x, y;

  memset(green_supported, 0, (size_t)w * h);

  for (y = 0U; y < h; y++) {
    for (x = 0U; x < w; x++) {
      uint8_t support = 0U;
      int16_t xmin = (int16_t)x - CCRD_GREEN_EDGE_EXPAND_PX;
      int16_t xmax = (int16_t)x + CCRD_GREEN_EDGE_EXPAND_PX;
      int16_t xx;

      if (xmin < 0) xmin = 0;
      if (xmax >= (int16_t)w) xmax = (int16_t)w - 1;

      for (xx = xmin; xx <= xmax; xx++) {
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
  uint16_t nobj = 0U;
  uint16_t i;

  build_green_supported(img);
  ncomp = connected_components(img, green_supported, comps);

  for (i = 0U; i < ncomp; i++) {
    uint16_t sx1, sy1, sx2, sy2;
    uint16_t x1e, y1e, x2e, y2e;
    uint16_t roi_w, roi_h;
    uint32_t roi_area;
    uint32_t raw_green_cnt = 0U;
    int16_t first_row, last_row, first_col, last_col;
    uint16_t gx1, gx2, gy1, gy2;
    uint16_t bw, bh;
    uint32_t box_area;
    uint32_t raw_cnt = 0U, supp_cnt = 0U;
    uint16_t x, y;

    if (comps[i].area < CCRD_GREEN_MIN_SUPPORTED_PIXELS) {
      continue;
    }

    sx1 = comps[i].x1; sy1 = comps[i].y1; sx2 = comps[i].x2; sy2 = comps[i].y2;
    x1e = (sx1 > CCRD_GREEN_BBOX_EXPAND_X) ? (uint16_t)(sx1 - CCRD_GREEN_BBOX_EXPAND_X) : 0U;
    y1e = (sy1 > CCRD_GREEN_BBOX_EXPAND_Y) ? (uint16_t)(sy1 - CCRD_GREEN_BBOX_EXPAND_Y) : 0U;
    x2e = (uint16_t)(((uint32_t)sx2 + CCRD_GREEN_BBOX_EXPAND_X < w) ? (sx2 + CCRD_GREEN_BBOX_EXPAND_X) : w);
    y2e = (uint16_t)(((uint32_t)sy2 + CCRD_GREEN_BBOX_EXPAND_Y < h) ? (sy2 + CCRD_GREEN_BBOX_EXPAND_Y) : h);

    roi_w = (uint16_t)(x2e - x1e);
    roi_h = (uint16_t)(y2e - y1e);
    roi_area = (uint32_t)roi_w * (uint32_t)roi_h;

    if (roi_h > CCRD_MAX_DIM || roi_w > CCRD_MAX_DIM) {
      continue;
    }

    memset(row_counts, 0, (size_t)roi_h * sizeof(uint16_t));
    memset(col_counts, 0, (size_t)roi_w * sizeof(uint16_t));

    for (y = y1e; y < y2e; y++) {
      for (x = x1e; x < x2e; x++) {
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

    first_row = -1; last_row = -1; first_col = -1; last_col = -1;

    for (y = 0U; y < roi_h; y++) {
      if ((uint32_t)row_counts[y] * CCRD_GREEN_ROW_DENSITY_DEN >=
          (uint32_t)roi_w * CCRD_GREEN_ROW_DENSITY_NUM) {
        if (first_row < 0) first_row = (int16_t)y;
        last_row = (int16_t)y;
      }
    }

    for (x = 0U; x < roi_w; x++) {
      if ((uint32_t)col_counts[x] * CCRD_GREEN_COL_DENSITY_DEN >=
          (uint32_t)roi_h * CCRD_GREEN_COL_DENSITY_NUM) {
        if (first_col < 0) first_col = (int16_t)x;
        last_col = (int16_t)x;
      }
    }

    if (first_row < 0 || first_col < 0) {
      continue;
    }

    gx1 = (uint16_t)(x1e + first_col);
    gx2 = (uint16_t)(x1e + last_col + 1);
    gy1 = (uint16_t)(y1e + first_row);
    gy2 = (uint16_t)(y1e + last_row + 1);
    bw = (uint16_t)(gx2 - gx1);
    bh = (uint16_t)(gy2 - gy1);
    box_area = (uint32_t)bw * (uint32_t)bh;

    if (box_area < CCRD_GREEN_MIN_BOX_AREA) continue;
    if (bh < CCRD_GREEN_MIN_HEIGHT) continue;
    if ((uint32_t)bh * CCRD_GREEN_MIN_ASPECT_DEN < (uint32_t)bw * CCRD_GREEN_MIN_ASPECT_NUM) continue;

    for (y = gy1; y < gy2; y++) {
      for (x = gx1; x < gx2; x++) {
        raw_cnt  += green_mask[idx_of(w, x, y)] ? 1U : 0U;
        supp_cnt += green_supported[idx_of(w, x, y)] ? 1U : 0U;
      }
    }

    if (!ratio_ge_u32(raw_cnt, box_area,
                      CCRD_GREEN_MIN_RAW_DENSITY_NUM,
                      CCRD_GREEN_MIN_RAW_DENSITY_DEN)) {
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

    for (y = gy1; y < gy2; y++) {
      for (x = gx1; x < gx2; x++) {
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
  uint8_t i;

  result->left = 0U;
  result->middle = 0U;
  result->right = 0U;

  for (i = 0U; i < 3U; i++) {
    uint16_t x1 = col_edges[i];
    uint16_t x2 = col_edges[i + 1U];
    uint32_t orange_count = 0U;
    uint32_t green_count = 0U;
    uint32_t orange_total = (uint32_t)(x2 - x1) * (uint32_t)(orange_y2 - orange_y1);
    uint32_t green_total  = (uint32_t)(x2 - x1) * (uint32_t)(green_y2 - green_y1);
    uint16_t x, y;

    for (y = orange_y1; y < orange_y2; y++) {
      for (x = x1; x < x2; x++) {
        orange_count += orange_valid[idx_of(w, x, y)] ? 1U : 0U;
      }
    }
    for (y = green_y1; y < green_y2; y++) {
      for (x = x1; x < x2; x++) {
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
                  result->metrics[i].orange_blocked ? 200U : 120U);
      draw_rect_y((struct image_t *)img, x1, green_y1, x2, green_y2,
                  result->metrics[i].green_blocked ? 255U : 80U);
      draw_rect_y((struct image_t *)img, x1, 0U, x2, h,
                  result->metrics[i].blocked ? 255U : 180U);
    }
  }

  result->left   = result->metrics[0].blocked ? 1U : 0U;
  result->middle = result->metrics[1].blocked ? 1U : 0U;
  result->right  = result->metrics[2].blocked ? 1U : 0U;
}

static struct image_t *custom_color_risk_detector_func(struct image_t *img,
                                                       uint8_t camera_id __attribute__((unused)))
{
  struct orange_band_t bands[CCRD_MAX_ORANGE_BANDS];
  struct bbox_t orange_objects[CCRD_MAX_ORANGE_OBJECTS];
  struct bbox_t green_objects[CCRD_MAX_GREEN_OBJECTS];
  struct risk_result_t local_result;
  struct image_t *proc = img;
  uint32_t input_npixels = (uint32_t)img->w * (uint32_t)img->h;
  uint64_t t0, t1, t2, t3, t4, t5, t6, t7;
  uint16_t nbands, norange, ngreen;

  if (img->type != IMAGE_YUV422) {
    return img;
  }
  if (input_npixels > CCRD_MAX_PIXELS) {
    return img;
  }

  t0 = now_us();
  memset(&local_result, 0, sizeof(local_result));

#if CCRD_ENABLE_DOWNSAMPLE
  {
    uint8_t ds = CCRD_DOWNSAMPLE;
    uint32_t out_w, out_h, out_bytes;

    if (ds < 1U) {
      ds = 1U;
    }

    out_w = img->w / ds;
    out_h = img->h / ds;
    out_bytes = out_w * out_h * 2U;

    if (ds > 1U && out_bytes <= CCRD_MAX_DOWNSAMPLED_BYTES) {
      memset(&downsampled_img, 0, sizeof(downsampled_img));
      downsampled_img.buf = downsample_buf;
      image_yuv422_downsample(img, &downsampled_img, ds);
      proc = &downsampled_img;
    }
  }
#endif

  t1 = now_us();
  build_masks(proc);
  t2 = now_us();

  nbands = scan_orange_vertical_bands(proc, bands);
  t3 = now_us();

  norange = group_orange_bands(proc, bands, nbands, orange_objects);
  t4 = now_us();

  ngreen = detect_green_objects(proc, green_objects);
  t5 = now_us();

  compute_risk_map(proc, &local_result);
  t6 = now_us();

  if (CUSTOM_COLOR_RISK_DETECTOR_DEBUG) {
    uint16_t i;
    draw_grid_debug(proc);
    for (i = 0U; i < norange; i++) {
      draw_rect_y(proc,
                  orange_objects[i].x1, orange_objects[i].y1,
                  orange_objects[i].x2, orange_objects[i].y2, 220U);
    }
    for (i = 0U; i < ngreen; i++) {
      draw_rect_y(proc,
                  green_objects[i].x1, green_objects[i].y1,
                  green_objects[i].x2, green_objects[i].y2, 150U);
    }
  }
  t7 = now_us();

  local_result.updated = true;

  pthread_mutex_lock(&ccrd_mutex);
  memcpy(&g_result, &local_result, sizeof(local_result));
  pthread_mutex_unlock(&ccrd_mutex);

  ccrd_frame_counter++;

  if ((ccrd_frame_counter % CCRD_LOG_EVERY_N_FRAMES) == 0U) {
    double ms_downsample   = (double)(t1 - t0) / 1000.0;
    double ms_build_masks  = (double)(t2 - t1) / 1000.0;
    double ms_scan_orange  = (double)(t3 - t2) / 1000.0;
    double ms_group_orange = (double)(t4 - t3) / 1000.0;
    double ms_detect_green = (double)(t5 - t4) / 1000.0;
    double ms_risk         = (double)(t6 - t5) / 1000.0;
    double ms_debug        = (double)(t7 - t6) / 1000.0;
    double ms_total        = (double)(t7 - t0) / 1000.0;

    VERBOSE_PRINT("frame=%lu ds=%u proc=%ux%u orange_objects=%u green_objects=%u risk=[%u,%u,%u]\n",
                  (unsigned long)ccrd_frame_counter,
                  (unsigned int)((proc == img) ? 1U : CCRD_DOWNSAMPLE),
                  proc->w, proc->h,
                  norange, ngreen,
                  local_result.left, local_result.middle, local_result.right);

    VERBOSE_PRINT("timing_ms frame=%lu downsample=%.3f masks=%.3f scan_orange=%.3f group_orange=%.3f detect_green=%.3f risk=%.3f debug=%.3f total=%.3f\n",
                  (unsigned long)ccrd_frame_counter,
                  ms_downsample,
                  ms_build_masks,
                  ms_scan_orange,
                  ms_group_orange,
                  ms_detect_green,
                  ms_risk,
                  ms_debug,
                  ms_total);
  }

  return img;
}

void custom_color_risk_detector_init(void)
{
  memset(&g_result, 0, sizeof(g_result));
  pthread_mutex_init(&ccrd_mutex, NULL);

  ccrd_open_log_file();
  ccrd_log("[custom_color_risk_detector->%s()] init\n", __FUNCTION__);

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