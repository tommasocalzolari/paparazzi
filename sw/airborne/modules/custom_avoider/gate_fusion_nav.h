/*
 * Fused navigation module:
 * - subscribes to CUSTOM_DETECTION from obstacle detector
 * - subscribes to VISUAL_DETECTION from gate CNN detector
 * - always keeps obstacle avoidance active
 * - follows the gate when visible
 * - when very close to the gate, commits to a short blind forward motion
 */

#include "modules/gate_fusion_nav/gate_fusion_nav.h"
#include "firmwares/rotorcraft/navigation.h"
#include "generated/airframe.h"
#include "state.h"
#include "modules/core/abi.h"

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "generated/flight_plan.h"

#ifndef GATE_FUSION_NAV_VERBOSE
#define GATE_FUSION_NAV_VERBOSE 1
#endif

#define PRINT(string,...) fprintf(stderr, "[gate_fusion_nav->%s()] " string, __FUNCTION__, ##__VA_ARGS__)
#if GATE_FUSION_NAV_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

/* --------------------------------------------------------- */
/* ABI subscriptions                                         */
/* --------------------------------------------------------- */

#ifndef GATE_FUSION_CUSTOM_DETECTION_ID
#define GATE_FUSION_CUSTOM_DETECTION_ID ABI_BROADCAST
#endif

#ifndef GATE_FUSION_VISUAL_DETECTION_ID
#define GATE_FUSION_VISUAL_DETECTION_ID ABI_BROADCAST
#endif

static abi_event obstacle_ev;
static abi_event gate_ev;

/* obstacle detector output */
static uint8_t objectLeft = 0;
static uint8_t objectMiddle = 0;
static uint8_t objectRight = 0;

/* gate detector output */
static uint8_t gate_detected = 0;
static int16_t gate_px = 0;
static int16_t gate_py = 0;
static int16_t gate_pw = 0;
static int16_t gate_ph = 0;
static int32_t gate_quality = 0;

/* --------------------------------------------------------- */
/* Tunable settings                                          */
/* --------------------------------------------------------- */

#ifndef GATE_FUSION_IMAGE_CENTER_X
#define GATE_FUSION_IMAGE_CENTER_X 120
#endif

#ifndef GATE_FUSION_ALIGN_TOL_PX
#define GATE_FUSION_ALIGN_TOL_PX 20
#endif

#ifndef GATE_FUSION_MIN_QUALITY
#define GATE_FUSION_MIN_QUALITY 250
#endif

#ifndef GATE_FUSION_BLIND_QUALITY
#define GATE_FUSION_BLIND_QUALITY 500
#endif

#ifndef GATE_FUSION_BLIND_MIN_WIDTH
#define GATE_FUSION_BLIND_MIN_WIDTH 120
#endif

#ifndef GATE_FUSION_BLIND_MIN_HEIGHT
#define GATE_FUSION_BLIND_MIN_HEIGHT 120
#endif

#ifndef GATE_FUSION_BLIND_CYCLES
#define GATE_FUSION_BLIND_CYCLES 10
#endif

#ifndef GATE_FUSION_FORWARD_DIST
#define GATE_FUSION_FORWARD_DIST 1.0f
#endif

#ifndef GATE_FUSION_TRAJ_FORWARD_DIST
#define GATE_FUSION_TRAJ_FORWARD_DIST 1.5f
#endif

#ifndef GATE_FUSION_SEARCH_HEADING_INC_DEG
#define GATE_FUSION_SEARCH_HEADING_INC_DEG 5.0f
#endif

#ifndef GATE_FUSION_ALIGN_HEADING_INC_DEG
#define GATE_FUSION_ALIGN_HEADING_INC_DEG 4.0f
#endif

#ifndef GATE_FUSION_AVOID_HEADING_INC_DEG
#define GATE_FUSION_AVOID_HEADING_INC_DEG 6.0f
#endif

#ifndef GATE_FUSION_MAX_DISTANCE
#define GATE_FUSION_MAX_DISTANCE 2.25f
#endif

/* --------------------------------------------------------- */
/* State machine                                             */
/* --------------------------------------------------------- */

enum navigation_state_t {
  SEARCH_FOR_GATE = 0,
  ALIGN_TO_GATE,
  FLY_TO_GATE,
  BLIND_THROUGH_GATE,
  OUT_OF_BOUNDS
};

static enum navigation_state_t navigation_state = SEARCH_FOR_GATE;
static float heading_increment = 5.f;
static uint8_t blind_cycles_remaining = 0;

/* --------------------------------------------------------- */
/* Function declarations                                     */
/* --------------------------------------------------------- */

static uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters);
static uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters);
static uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor);
static uint8_t increase_nav_heading(float incrementDegrees);
static uint8_t chooseRandomIncrementAvoidance(void);
static void stop_waypoints_here(void);
static uint8_t gate_is_good(void);
static uint8_t gate_is_close(void);

/* --------------------------------------------------------- */
/* ABI callbacks                                             */
/* --------------------------------------------------------- */

static void object_detection_cb(uint8_t sender_id __attribute__((unused)),
                                uint8_t left,
                                uint8_t middle,
                                uint8_t right)
{
  objectLeft = left;
  objectMiddle = middle;
  objectRight = right;
}

static void gate_detection_cb(uint8_t sender_id __attribute__((unused)),
                              int16_t pixel_x,
                              int16_t pixel_y,
                              int16_t pixel_width,
                              int16_t pixel_height,
                              int32_t quality,
                              int16_t extra __attribute__((unused)))
{
  gate_px = pixel_x;
  gate_py = pixel_y;
  gate_pw = pixel_width;
  gate_ph = pixel_height;
  gate_quality = quality;
  gate_detected = (quality > 0) ? 1 : 0;
}

/* --------------------------------------------------------- */
/* Init                                                      */
/* --------------------------------------------------------- */

void gate_fusion_nav_init(void)
{
  srand(time(NULL));
  chooseRandomIncrementAvoidance();
  blind_cycles_remaining = 0;
  navigation_state = SEARCH_FOR_GATE;

  AbiBindMsgCUSTOM_DETECTION(GATE_FUSION_CUSTOM_DETECTION_ID,
                             &obstacle_ev,
                             object_detection_cb);

  AbiBindMsgVISUAL_DETECTION(GATE_FUSION_VISUAL_DETECTION_ID,
                             &gate_ev,
                             gate_detection_cb);
}

/* --------------------------------------------------------- */
/* Main periodic logic                                       */
/* --------------------------------------------------------- */

void gate_fusion_nav_periodic(void)
{
  if (!autopilot_in_flight()) {
    return;
  }

  VERBOSE_PRINT("state=%d obs[L,M,R]=[%d,%d,%d] gate=%d px=%d pw=%d ph=%d q=%ld blind=%u\n",
                navigation_state,
                objectLeft, objectMiddle, objectRight,
                gate_detected, gate_px, gate_pw, gate_ph,
                (long)gate_quality, blind_cycles_remaining);

  switch (navigation_state) {

    case SEARCH_FOR_GATE:
      stop_waypoints_here();

      /* middle obstacle has highest priority */
      if (objectMiddle) {
        increase_nav_heading(heading_increment);
        break;
      }

      if (gate_is_good()) {
        navigation_state = ALIGN_TO_GATE;
        VERBOSE_PRINT("State: ALIGN_TO_GATE\n");
      } else {
        /* old search behavior */
        increase_nav_heading(heading_increment);
      }
      break;

    case ALIGN_TO_GATE: {
      stop_waypoints_here();

      if (objectMiddle) {
        navigation_state = SEARCH_FOR_GATE;
        chooseRandomIncrementAvoidance();
        VERBOSE_PRINT("State: SEARCH_FOR_GATE (middle obstacle)\n");
        break;
      }

      if (!gate_is_good()) {
        navigation_state = SEARCH_FOR_GATE;
        VERBOSE_PRINT("State: SEARCH_FOR_GATE (gate lost)\n");
        break;
      }

      int16_t err_x = gate_px - GATE_FUSION_IMAGE_CENTER_X;

      /* obstacle side-bias always active */
      if (objectLeft && !objectRight) {
        increase_nav_heading(GATE_FUSION_AVOID_HEADING_INC_DEG);
      } else if (objectRight && !objectLeft) {
        increase_nav_heading(-GATE_FUSION_AVOID_HEADING_INC_DEG);
      } else {
        if (err_x < -GATE_FUSION_ALIGN_TOL_PX) {
          /* gate is left in image -> yaw left */
          increase_nav_heading(GATE_FUSION_ALIGN_HEADING_INC_DEG);
        } else if (err_x > GATE_FUSION_ALIGN_TOL_PX) {
          /* gate is right in image -> yaw right */
          increase_nav_heading(-GATE_FUSION_ALIGN_HEADING_INC_DEG);
        } else {
          navigation_state = FLY_TO_GATE;
          VERBOSE_PRINT("State: FLY_TO_GATE\n");
        }
      }
      break;
    }

    case FLY_TO_GATE:
      if (objectMiddle) {
        stop_waypoints_here();
        chooseRandomIncrementAvoidance();
        navigation_state = SEARCH_FOR_GATE;
        VERBOSE_PRINT("State: SEARCH_FOR_GATE (middle obstacle while flying)\n");
        break;
      }

      if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
        navigation_state = OUT_OF_BOUNDS;
        VERBOSE_PRINT("State: OUT_OF_BOUNDS\n");
        break;
      }

      if (!gate_is_good()) {
        navigation_state = SEARCH_FOR_GATE;
        VERBOSE_PRINT("State: SEARCH_FOR_GATE (gate lost while flying)\n");
        break;
      }

      /* if close to gate: commit */
      if (gate_is_close()) {
        blind_cycles_remaining = GATE_FUSION_BLIND_CYCLES;
        navigation_state = BLIND_THROUGH_GATE;
        VERBOSE_PRINT("State: BLIND_THROUGH_GATE\n");
        break;
      }

      /* mild side obstacle bias but keep going */
      if (objectLeft && !objectRight) {
        increase_nav_heading(GATE_FUSION_AVOID_HEADING_INC_DEG);
      } else if (objectRight && !objectLeft) {
        increase_nav_heading(-GATE_FUSION_AVOID_HEADING_INC_DEG);
      } else {
        int16_t err_x = gate_px - GATE_FUSION_IMAGE_CENTER_X;
        if (err_x < -GATE_FUSION_ALIGN_TOL_PX) {
          increase_nav_heading(GATE_FUSION_ALIGN_HEADING_INC_DEG);
        } else if (err_x > GATE_FUSION_ALIGN_TOL_PX) {
          increase_nav_heading(-GATE_FUSION_ALIGN_HEADING_INC_DEG);
        }
      }

      moveWaypointForward(WP_TRAJECTORY, GATE_FUSION_TRAJ_FORWARD_DIST);
      moveWaypointForward(WP_GOAL, GATE_FUSION_FORWARD_DIST);
      break;

    case BLIND_THROUGH_GATE:
      /*
       * Short committed forward pass.
       * Ignore gate loss here; still respect middle obstacle as emergency escape.
       */
      if (objectMiddle) {
        stop_waypoints_here();
        chooseRandomIncrementAvoidance();
        navigation_state = SEARCH_FOR_GATE;
        blind_cycles_remaining = 0;
        VERBOSE_PRINT("State: SEARCH_FOR_GATE (middle obstacle during blind pass)\n");
        break;
      }

      moveWaypointForward(WP_TRAJECTORY, GATE_FUSION_TRAJ_FORWARD_DIST);
      moveWaypointForward(WP_GOAL, GATE_FUSION_FORWARD_DIST);

      if (blind_cycles_remaining > 0) {
        blind_cycles_remaining--;
      }

      if (blind_cycles_remaining == 0) {
        navigation_state = SEARCH_FOR_GATE;
        VERBOSE_PRINT("State: SEARCH_FOR_GATE (blind pass complete)\n");
      }
      break;

    case OUT_OF_BOUNDS:
      increase_nav_heading(heading_increment);
      moveWaypointForward(WP_TRAJECTORY, 1.5f);

      if (InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
        increase_nav_heading(heading_increment);
        navigation_state = SEARCH_FOR_GATE;
        VERBOSE_PRINT("State: SEARCH_FOR_GATE\n");
      }
      break;

    default:
      navigation_state = SEARCH_FOR_GATE;
      break;
  }
}

/* --------------------------------------------------------- */
/* Helpers                                                   */
/* --------------------------------------------------------- */

static void stop_waypoints_here(void)
{
  waypoint_move_here_2d(WP_GOAL);
  waypoint_move_here_2d(WP_TRAJECTORY);
}

static uint8_t gate_is_good(void)
{
  return (gate_detected && gate_quality >= GATE_FUSION_MIN_QUALITY) ? 1U : 0U;
}

static uint8_t gate_is_close(void)
{
  if (!gate_detected) {
    return 0U;
  }

  if (gate_quality >= GATE_FUSION_BLIND_QUALITY) {
    return 1U;
  }

  if (gate_pw >= GATE_FUSION_BLIND_MIN_WIDTH) {
    return 1U;
  }

  if (gate_ph >= GATE_FUSION_BLIND_MIN_HEIGHT) {
    return 1U;
  }

  return 0U;
}

static uint8_t increase_nav_heading(float incrementDegrees)
{
  float new_heading = stateGetNedToBodyEulers_f()->psi + RadOfDeg(incrementDegrees);
  FLOAT_ANGLE_NORMALIZE(new_heading);
  nav.heading = new_heading;
  return false;
}

static uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters)
{
  struct EnuCoor_i new_coor;
  calculateForwards(&new_coor, distanceMeters);
  moveWaypoint(waypoint, &new_coor);
  return false;
}

static uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters)
{
  float heading = stateGetNedToBodyEulers_f()->psi;

  new_coor->x = stateGetPositionEnu_i()->x + POS_BFP_OF_REAL(sinf(heading) * distanceMeters);
  new_coor->y = stateGetPositionEnu_i()->y + POS_BFP_OF_REAL(cosf(heading) * distanceMeters);
  return false;
}

static uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor)
{
  waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);
  return false;
}

static uint8_t chooseRandomIncrementAvoidance(void)
{
  if (rand() % 2 == 0) {
    heading_increment = GATE_FUSION_SEARCH_HEADING_INC_DEG;
  } else {
    heading_increment = -GATE_FUSION_SEARCH_HEADING_INC_DEG;
  }
  return false;
}