/*
 * Copyright (C) Roland Meertens
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/custom_avoider/custom_avoider.c"
 * Custom avoider with orange_avoider-like confidence-based forward motion.
 */

#include "modules/custom_avoider/custom_avoider.h"
#include "firmwares/rotorcraft/navigation.h"
#include "generated/airframe.h"
#include "state.h"
#include "modules/core/abi.h"
#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <stdarg.h>
#include <string.h>
#include <sys/time.h>
#include <pthread.h>

#include "generated/flight_plan.h"

#define CUSTOM_AVOIDER_VERBOSE TRUE

#define PRINT(string,...) fprintf(stderr, "[custom_avoider->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if CUSTOM_AVOIDER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

/* -------------------------------------------------------------------------- */
/* Profiling / logging                                                        */
/* -------------------------------------------------------------------------- */

#ifndef CUSTOM_AVOIDER_PROFILE
#define CUSTOM_AVOIDER_PROFILE 1
#endif

#ifndef CUSTOM_AVOIDER_PROFILE_LOG_PATH
#define CUSTOM_AVOIDER_PROFILE_LOG_PATH "/tmp/custom_avoider_timing.log"
#endif

#ifndef CUSTOM_AVOIDER_PROFILE_FLUSH_EVERY_LINE
#define CUSTOM_AVOIDER_PROFILE_FLUSH_EVERY_LINE 1
#endif

#ifndef CUSTOM_AVOIDER_LOG_EVERY_N_PERIODIC
#define CUSTOM_AVOIDER_LOG_EVERY_N_PERIODIC 1
#endif

static pthread_mutex_t custom_avoider_log_mutex;
static FILE *custom_avoider_log_file = NULL;
static uint32_t custom_avoider_periodic_counter = 0;

static inline uint64_t ca_time_now_us(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ((uint64_t)ts.tv_sec * 1000000ULL) + ((uint64_t)ts.tv_nsec / 1000ULL);
}

static void ca_open_log_file(void)
{
#if CUSTOM_AVOIDER_PROFILE
  pthread_mutex_lock(&custom_avoider_log_mutex);

  if (custom_avoider_log_file == NULL) {
    custom_avoider_log_file = fopen(CUSTOM_AVOIDER_PROFILE_LOG_PATH, "a");
    if (custom_avoider_log_file != NULL) {
      setvbuf(custom_avoider_log_file, NULL, _IOLBF, 0);
      fprintf(custom_avoider_log_file,
              "\n===== custom_avoider profiling started =====\n");
    } else {
      fprintf(stderr,
              "[custom_avoider] Failed to open log file: %s\n",
              CUSTOM_AVOIDER_PROFILE_LOG_PATH);
    }
  }

  pthread_mutex_unlock(&custom_avoider_log_mutex);
#endif
}

static void ca_log_simple(const char *func_name, uint64_t dt_us)
{
#if CUSTOM_AVOIDER_PROFILE
  pthread_mutex_lock(&custom_avoider_log_mutex);

  if (custom_avoider_log_file != NULL) {
    struct timeval tv;
    gettimeofday(&tv, NULL);

    fprintf(custom_avoider_log_file,
            "%ld.%06ld,%s,%llu us\n",
            (long)tv.tv_sec,
            (long)tv.tv_usec,
            func_name,
            (unsigned long long)dt_us);

#if CUSTOM_AVOIDER_PROFILE_FLUSH_EVERY_LINE
    fflush(custom_avoider_log_file);
#endif
  }

  pthread_mutex_unlock(&custom_avoider_log_mutex);
#else
  (void)func_name;
  (void)dt_us;
#endif
}

static void ca_log_callback(uint64_t dt_us, uint8_t left, uint8_t middle, uint8_t right)
{
#if CUSTOM_AVOIDER_PROFILE
  pthread_mutex_lock(&custom_avoider_log_mutex);

  if (custom_avoider_log_file != NULL) {
    struct timeval tv;
    gettimeofday(&tv, NULL);

    fprintf(custom_avoider_log_file,
            "%ld.%06ld,object_detection_cb,%llu us,left=%u,middle=%u,right=%u\n",
            (long)tv.tv_sec,
            (long)tv.tv_usec,
            (unsigned long long)dt_us,
            left, middle, right);

#if CUSTOM_AVOIDER_PROFILE_FLUSH_EVERY_LINE
    fflush(custom_avoider_log_file);
#endif
  }

  pthread_mutex_unlock(&custom_avoider_log_mutex);
#else
  (void)dt_us;
  (void)left;
  (void)middle;
  (void)right;
#endif
}

static void ca_log_periodic(uint64_t dt_us,
                            int state_before,
                            int state_after,
                            uint8_t left,
                            uint8_t middle,
                            uint8_t right,
                            int16_t leftAccum,
                            int16_t middleAccum,
                            int16_t rightAccum,
                            int16_t free_conf,
                            float moveDistance)
{
#if CUSTOM_AVOIDER_PROFILE
  pthread_mutex_lock(&custom_avoider_log_mutex);

  if (custom_avoider_log_file != NULL) {
    struct timeval tv;
    gettimeofday(&tv, NULL);

    fprintf(custom_avoider_log_file,
            "%ld.%06ld,navigation_controller_periodic,%llu us,state_before=%d,state_after=%d,left=%u,middle=%u,right=%u,leftAccum=%d,middleAccum=%d,rightAccum=%d,free_conf=%d,moveDistance=%.2f\n",
            (long)tv.tv_sec,
            (long)tv.tv_usec,
            (unsigned long long)dt_us,
            state_before,
            state_after,
            left, middle, right,
            leftAccum, middleAccum, rightAccum,
            free_conf,
            moveDistance);

#if CUSTOM_AVOIDER_PROFILE_FLUSH_EVERY_LINE
    fflush(custom_avoider_log_file);
#endif
  }

  pthread_mutex_unlock(&custom_avoider_log_mutex);
#else
  (void)dt_us;
  (void)state_before;
  (void)state_after;
  (void)left;
  (void)middle;
  (void)right;
  (void)leftAccum;
  (void)middleAccum;
  (void)rightAccum;
  (void)free_conf;
  (void)moveDistance;
#endif
}

/* -------------------------------------------------------------------------- */

static uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters);
static uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters);
static uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor);
static uint8_t increase_nav_heading(float incrementDegrees);
static uint8_t chooseRandomIncrementAvoidance(void);

static inline int16_t clamp_i16(int16_t v, int16_t vmin, int16_t vmax)
{
  if (v < vmin) { return vmin; }
  if (v > vmax) { return vmax; }
  return v;
}

enum navigation_state_t {
  SAFE,
  OBSTACLE_LEFT,
  OBSTACLE_RIGHT,
  OBSTACLE_MIDDLE,
  SEARCH_FOR_SAFE_HEADING,
  OUT_OF_BOUNDS
};

// -----------------------------------------------------------------------------
// Global state
// -----------------------------------------------------------------------------

enum navigation_state_t navigation_state = SEARCH_FOR_SAFE_HEADING;

volatile uint8_t objectLeft = 0;
volatile uint8_t objectMiddle = 0;
volatile uint8_t objectRight = 0;

// Signed accumulators for temporal filtering
int16_t objectLeftAccum = 0;
int16_t objectMiddleAccum = 0;
int16_t objectRightAccum = 0;

// Filtering parameters
int16_t confidenceThreshold = 2; // lower to make it more reactive :  init value = 3
int16_t confidenceMax = 3;       // lower to make it more reactive :  init value = 6

// Orange-avoider-like confidence for forward motion
int16_t obstacle_free_confidence = 0;
const int16_t max_trajectory_confidence = 5;

// Turning and motion parameters
float heading_increment = 5.f;   // used in search mode
float maxDistance = 2.25f;       // cap for forward motion

#ifndef CUSTOM_AVOIDER_CUSTOM_DETECTION_ID
#define CUSTOM_AVOIDER_CUSTOM_DETECTION_ID ABI_BROADCAST
#endif

static abi_event color_detection_ev;

static void object_detection_cb(uint8_t __attribute__((unused)) sender_id,
                                uint8_t left,
                                uint8_t middle,
                                uint8_t right)
{
  uint64_t t0 = ca_time_now_us();

  objectLeft = left;
  objectMiddle = middle;
  objectRight = right;

  ca_log_callback(ca_time_now_us() - t0, left, middle, right);
}

void navigation_controller_init(void)
{
  uint64_t t0 = ca_time_now_us();

  pthread_mutex_init(&custom_avoider_log_mutex, NULL);
  ca_open_log_file();

  srand(time(NULL));
  chooseRandomIncrementAvoidance();

  AbiBindMsgCUSTOM_DETECTION(CUSTOM_AVOIDER_CUSTOM_DETECTION_ID,
                             &color_detection_ev,
                             object_detection_cb);

  ca_log_simple("navigation_controller_init", ca_time_now_us() - t0);
}

void navigation_controller_periodic(void)
{
  uint64_t t0 = ca_time_now_us();
  int state_before = (int)navigation_state;

  if (!autopilot_in_flight()) {
    ca_log_simple("navigation_controller_periodic_not_flying", ca_time_now_us() - t0);
    return;
  }

  // Side obstacle local turning increment
  float headingIncrement = 1.0f;

  // ---------------------------------------------------------------------------
  // Update per-sector confidence accumulators
  // ---------------------------------------------------------------------------
  objectLeftAccum   += (objectLeft   ? 1 : -1);
  objectMiddleAccum += (objectMiddle ? 1 : -1);
  objectRightAccum  += (objectRight  ? 1 : -1);

  objectLeftAccum   = clamp_i16(objectLeftAccum,   0, confidenceMax);
  objectMiddleAccum = clamp_i16(objectMiddleAccum, 0, confidenceMax);
  objectRightAccum  = clamp_i16(objectRightAccum,  0, confidenceMax);

  bool leftConfirmed   = (objectLeftAccum   >= confidenceThreshold);
  bool middleConfirmed = (objectMiddleAccum >= confidenceThreshold);
  bool rightConfirmed  = (objectRightAccum  >= confidenceThreshold);

  bool anyObstacleConfirmed = leftConfirmed || middleConfirmed || rightConfirmed;

  // ---------------------------------------------------------------------------
  // Orange-avoider-like free-path confidence update
  // ---------------------------------------------------------------------------
  if (!anyObstacleConfirmed) {
    obstacle_free_confidence += 2;   // it was +1
  } else {
    obstacle_free_confidence--;  // it was -2
  }

  obstacle_free_confidence = clamp_i16(obstacle_free_confidence, 0, max_trajectory_confidence);

  // Forward distance now depends on confidence, like orange_avoider
  float moveDistance = fminf(maxDistance, 0.2f * obstacle_free_confidence);

  VERBOSE_PRINT("raw L/M/R = %u %u %u | accum L/M/R = %d %d %d | conf L/M/R = %d %d %d | free_conf = %d | moveDistance = %.2f | state = %d\n",
                objectLeft, objectMiddle, objectRight,
                objectLeftAccum, objectMiddleAccum, objectRightAccum,
                leftConfirmed, middleConfirmed, rightConfirmed,
                obstacle_free_confidence, moveDistance, navigation_state);

  switch (navigation_state) {

    case SAFE:
      moveWaypointForward(WP_TRAJECTORY, 1.5f * moveDistance);

      if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
        navigation_state = OUT_OF_BOUNDS;
      } else if (obstacle_free_confidence == 0) {
        if (middleConfirmed) {
          navigation_state = OBSTACLE_MIDDLE;
        } else if (leftConfirmed && !rightConfirmed) {
          navigation_state = OBSTACLE_LEFT;
        } else if (rightConfirmed && !leftConfirmed) {
          navigation_state = OBSTACLE_RIGHT;
        } 
        //else {
        //  navigation_state = OBSTACLE_MIDDLE;
        //}
      } else {
        moveWaypointForward(WP_GOAL, moveDistance);
      }
      break;

    case OBSTACLE_LEFT:
      waypoint_move_here_2d(WP_GOAL);
      waypoint_move_here_2d(WP_TRAJECTORY);

      increase_nav_heading(headingIncrement);

      if (!leftConfirmed && obstacle_free_confidence >= 1) {
        navigation_state = SAFE;
      }
      break;

    case OBSTACLE_RIGHT:
      waypoint_move_here_2d(WP_GOAL);
      waypoint_move_here_2d(WP_TRAJECTORY);

      increase_nav_heading(-headingIncrement);

      if (!rightConfirmed && obstacle_free_confidence >= 1) {
        navigation_state = SAFE;
      }
      break;

    case OBSTACLE_MIDDLE:
      waypoint_move_here_2d(WP_GOAL);
      waypoint_move_here_2d(WP_TRAJECTORY);

      chooseRandomIncrementAvoidance();
      navigation_state = SEARCH_FOR_SAFE_HEADING;
      break;

    case SEARCH_FOR_SAFE_HEADING:
      increase_nav_heading(heading_increment);

    if (obstacle_free_confidence >= 2 && !middleConfirmed) {
          navigation_state = SAFE;
        }
      break;

    case OUT_OF_BOUNDS:
      increase_nav_heading(heading_increment);
      moveWaypointForward(WP_TRAJECTORY, 1.5f);

      if (InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
        increase_nav_heading(heading_increment);

        obstacle_free_confidence = 0;
        navigation_state = SEARCH_FOR_SAFE_HEADING;
      }
      break;

    default:
      break;
  }

  custom_avoider_periodic_counter++;
  if ((custom_avoider_periodic_counter % CUSTOM_AVOIDER_LOG_EVERY_N_PERIODIC) == 0U) {
    ca_log_periodic(ca_time_now_us() - t0,
                    state_before,
                    (int)navigation_state,
                    objectLeft, objectMiddle, objectRight,
                    objectLeftAccum, objectMiddleAccum, objectRightAccum,
                    obstacle_free_confidence,
                    moveDistance);
  }
}

uint8_t increase_nav_heading(float incrementDegrees)
{
  uint64_t t0 = ca_time_now_us();

  float new_heading = stateGetNedToBodyEulers_f()->psi + RadOfDeg(incrementDegrees);

  FLOAT_ANGLE_NORMALIZE(new_heading);
  nav.heading = new_heading;

  ca_log_simple("increase_nav_heading", ca_time_now_us() - t0);
  return false;
}

uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters)
{
  uint64_t t0 = ca_time_now_us();

  struct EnuCoor_i new_coor;
  calculateForwards(&new_coor, distanceMeters);
  moveWaypoint(waypoint, &new_coor);

  ca_log_simple("moveWaypointForward", ca_time_now_us() - t0);
  return false;
}

uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters)
{
  uint64_t t0 = ca_time_now_us();

  float heading = stateGetNedToBodyEulers_f()->psi;

  new_coor->x = stateGetPositionEnu_i()->x + POS_BFP_OF_REAL(sinf(heading) * distanceMeters);
  new_coor->y = stateGetPositionEnu_i()->y + POS_BFP_OF_REAL(cosf(heading) * distanceMeters);

  ca_log_simple("calculateForwards", ca_time_now_us() - t0);
  return false;
}

uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor)
{
  uint64_t t0 = ca_time_now_us();

  waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);

  ca_log_simple("moveWaypoint", ca_time_now_us() - t0);
  return false;
}

uint8_t chooseRandomIncrementAvoidance(void)
{
  uint64_t t0 = ca_time_now_us();

  if (rand() % 2 == 0) {
    heading_increment = 5.f;
  } else {
    heading_increment = -5.f;
  }

  ca_log_simple("chooseRandomIncrementAvoidance", ca_time_now_us() - t0);
  return false;
}