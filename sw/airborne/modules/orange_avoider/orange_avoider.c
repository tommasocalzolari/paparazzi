/*
 * Orange avoider + Blue tracker (ABI-based, no raw image access)
 */

#include "modules/orange_avoider/orange_avoider.h"
#include "firmwares/rotorcraft/navigation.h"
#include "generated/airframe.h"
#include "state.h"
#include "modules/core/abi.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "generated/flight_plan.h"

#define ORANGE_AVOIDER_VERBOSE TRUE

#define IMAGE_WIDTH 240.0f
#define IMAGE_HEIGHT 520.0f

#define PRINT(string,...) fprintf(stderr, "[orange_avoider->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if ORANGE_AVOIDER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

// ===================== CONFIG =====================

#define ORANGE_AVOIDER_VISUAL_DETECTION_ID COLOR_OBJECT_DETECTION1_ID
#define BLUE_VISUAL_DETECTION_ID COLOR_OBJECT_DETECTION2_ID

// ===================== FUNCTION DECLARATIONS =====================
static float compute_heading_correction(void);
static uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters);
static uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters);
static uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor);
static uint8_t increase_nav_heading(float incrementDegrees);
static uint8_t chooseRandomIncrementAvoidance(void);
static uint8_t is_heading_aligned(float desired_correction);

// ===================== NAVIGATION =====================
enum navigation_state_t {
  SAFE,
  OBSTACLE_FOUND,
  SEARCH_FOR_SAFE_HEADING,
  OUT_OF_BOUNDS
};

// ===================== GLOBALS =====================

// Orange avoidance
float oa_color_count_frac = 0.18f;
enum navigation_state_t navigation_state = SAFE;
int32_t color_count = 0;
int16_t obstacle_free_confidence = 0;
float heading_increment = 5.f;
float maxDistance = 2.25;
const int16_t max_trajectory_confidence = 5;

// Blue tracking (ABI-based)
static int blue_seen = 0;
static int blue_cx = 0;
static int blue_cy = 0;

// ABI events
static abi_event orange_detection_ev;
static abi_event blue_detection_ev;

// ===================== ABI CALLBACKS =====================

// Orange (unchanged)
static void color_detection_cb(uint8_t sender_id,
                               int16_t pixel_x, int16_t pixel_y,
                               int16_t pixel_width, int16_t pixel_height,
                               int32_t quality, int16_t extra)
{
  color_count = quality;
}

// Blue (NEW)
static void blue_detection_cb(uint8_t sender_id,
                             int16_t pixel_x, int16_t pixel_y,
                             int16_t pixel_width, int16_t pixel_height,
                             int32_t quality, int16_t extra)
{
  if (quality > 0) {
    blue_seen = 1;
    blue_cx = pixel_x;
    blue_cy = pixel_y;
  } else {
    blue_seen = 0;
  }
}

// ===================== INIT =====================

void orange_avoider_init(void)
{
  srand(time(NULL));
  chooseRandomIncrementAvoidance();

  // Bind orange detection
  AbiBindMsgVISUAL_DETECTION(
    ORANGE_AVOIDER_VISUAL_DETECTION_ID,
    &orange_detection_ev,
    color_detection_cb
  );

  // Bind blue detection
  AbiBindMsgVISUAL_DETECTION(
    BLUE_VISUAL_DETECTION_ID,
    &blue_detection_ev,
    blue_detection_cb
  );
}

// ===================== BLUE STEERING =====================

float compute_heading_correction(void)
{
  if (!blue_seen) return 0.0f;

  float center_x = IMAGE_WIDTH / 2.0f;

  // normalized error [-1, 1]
  float error = (blue_cx - center_x) / center_x;

  float gain = 2.0f; // tune this
  return gain * error;
}

// ===================== MAIN LOOP =====================

void orange_avoider_periodic(void)
{
  if (!autopilot_in_flight()) return;

  // Threshold based on image size
  int32_t color_count_threshold =
    oa_color_count_frac *
    IMAGE_WIDTH *
    IMAGE_HEIGHT;

  VERBOSE_PRINT("Orange: %d  threshold: %d  state: %d\n",
                color_count, color_count_threshold, navigation_state);

  // Update confidence
  if(color_count < color_count_threshold){
    obstacle_free_confidence++;
  } else {
    obstacle_free_confidence -= 2;
  }

  Bound(obstacle_free_confidence, 0, max_trajectory_confidence);

  float moveDistance = fminf(maxDistance, 0.2f * obstacle_free_confidence);

  switch (navigation_state){

    case SAFE: {
        // Move forward if path is clear
        moveWaypointForward(WP_TRAJECTORY, 1.5f * moveDistance);

        if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY),WaypointY(WP_TRAJECTORY))){
            navigation_state = OUT_OF_BOUNDS;
        }
        else if (obstacle_free_confidence == 0){
            navigation_state = OBSTACLE_FOUND;
        }
        else {
            // 🔵 steer toward blue IF visible while moving
            if (blue_seen) {
                float correction = compute_heading_correction();
                increase_nav_heading(correction);
            }

            // Forward movement toward goal continues
            moveWaypointForward(WP_GOAL, moveDistance);
        }
    }
    break;

    case OBSTACLE_FOUND:
      waypoint_move_here_2d(WP_GOAL);
      waypoint_move_here_2d(WP_TRAJECTORY);
      chooseRandomIncrementAvoidance();
      navigation_state = SEARCH_FOR_SAFE_HEADING;
      break;

    case SEARCH_FOR_SAFE_HEADING:
      increase_nav_heading(heading_increment);
      if (obstacle_free_confidence >= 2){
        navigation_state = SAFE;
      }
      break;

    case OUT_OF_BOUNDS:
      increase_nav_heading(heading_increment);
      moveWaypointForward(WP_TRAJECTORY, 1.5f);

      if (InsideObstacleZone(WaypointX(WP_TRAJECTORY),WaypointY(WP_TRAJECTORY))){
        increase_nav_heading(heading_increment);
        obstacle_free_confidence = 0;
        navigation_state = SEARCH_FOR_SAFE_HEADING;
      }
      break;
  }
}

// ===================== NAV HELPERS =====================

uint8_t increase_nav_heading(float incrementDegrees)
{
  float new_heading =
    stateGetNedToBodyEulers_f()->psi +
    RadOfDeg(incrementDegrees);

  FLOAT_ANGLE_NORMALIZE(new_heading);
  nav.heading = new_heading;

  VERBOSE_PRINT("Heading: %f\n", DegOfRad(new_heading));
  return false;
}

uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters)
{
  struct EnuCoor_i new_coor;
  calculateForwards(&new_coor, distanceMeters);
  moveWaypoint(waypoint, &new_coor);
  return false;
}

uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters)
{
  float heading = stateGetNedToBodyEulers_f()->psi;

  new_coor->x =
    stateGetPositionEnu_i()->x +
    POS_BFP_OF_REAL(sinf(heading) * distanceMeters);

  new_coor->y =
    stateGetPositionEnu_i()->y +
    POS_BFP_OF_REAL(cosf(heading) * distanceMeters);

  return false;
}

uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor)
{
  waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);
  return false;
}

uint8_t chooseRandomIncrementAvoidance(void)
{
  heading_increment = (rand() % 2 == 0) ? 5.f : -5.f;
  return false;
}

// ===================== NEW HELPER =====================
static uint8_t is_heading_aligned(float desired_correction)
{
    return fabsf(desired_correction) < 2.0f; // within 2 degrees
}