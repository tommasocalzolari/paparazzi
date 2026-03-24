/*
 * Copyright (C) Roland Meertens
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/custom_avoider/custom_avoider.c"
 * @author Roland Meertens
 * Example on how to use the colours detected to avoid custom obstacle in the cyberzoo
 * This module is an example module for the course AE4317 Autonomous Flight of Micro Air Vehicles at the TU Delft.
 * This module is used in combination with a color filter (cv_detect_color_object) and the navigation mode of the autopilot.
 * The avoidance strategy is to simply count the total number of custom pixels. When above a certain percentage threshold,
 * (given by color_count_frac) we assume that there is an obstacle and we turn.
 *
 * The color filter settings are set using the cv_detect_color_object. This module can run multiple filters simultaneously
 * so you have to define which filter to use with the CUSTOM_AVOIDER_VISUAL_DETECTION_ID setting.
 */

#include "modules/custom_avoider/custom_avoider.h"
#include "firmwares/rotorcraft/navigation.h"
#include "generated/airframe.h"
#include "state.h"
#include "modules/core/abi.h"
#include "mcu_periph/sys_time.h"
#include <time.h>
#include <stdio.h>

#include "generated/flight_plan.h"

#define CUSTOM_AVOIDER_VERBOSE true

#define PRINT(string,...) fprintf(stderr, "[custom_avoider->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if CUSTOM_AVOIDER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

static uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters);
static uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters);
static uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor);
static uint8_t increase_nav_heading(float incrementDegrees);
static uint8_t chooseRandomIncrementAvoidance(void);
static void update_filtered_detection(uint8_t raw_detection, uint8_t *filtered_detection,
                                     uint8_t *candidate_detection, uint8_t *confirm_count,
                                     uint32_t *last_switch_ms, uint32_t now_ms);

enum navigation_state_t {
  SAFE,
  OBSTACLE_LEFT,
  OBSTACLE_RIGHT,
  OBSTACLE_MIDDLE,
  SEARCH_FOR_SAFE_HEADING,
  OUT_OF_BOUNDS
};

// define settings

// define and initialise global variables
enum navigation_state_t navigation_state = SEARCH_FOR_SAFE_HEADING;

uint8_t objectLeft = 0;
uint8_t objectMiddle = 0;
uint8_t objectRight = 0;

uint8_t objectLeftFiltered = 0;
uint8_t objectMiddleFiltered = 0;
uint8_t objectRightFiltered = 0;

uint8_t objectLeftCandidate = 0;
uint8_t objectMiddleCandidate = 0;
uint8_t objectRightCandidate = 0;

uint8_t objectLeftConfirmCount = 0;
uint8_t objectMiddleConfirmCount = 0;
uint8_t objectRightConfirmCount = 0;

uint32_t objectLeftLastSwitchMs = 0;
uint32_t objectMiddleLastSwitchMs = 0;
uint32_t objectRightLastSwitchMs = 0;

#ifndef CUSTOM_AVOIDER_CONFIRM_SAMPLES
#define CUSTOM_AVOIDER_CONFIRM_SAMPLES 3
#endif

#ifndef CUSTOM_AVOIDER_HOLD_TIME_MS
#define CUSTOM_AVOIDER_HOLD_TIME_MS 300U
#endif


float heading_increment = 1.f;          // heading angle increment [deg]
float maxDistance = 2.25;               // max waypoint displacement [m]
bool updateWaypoint = true;                // whether to update the waypoint position or just change heading when avoiding

/*
 * This next section defines an ABI messaging event (http://wiki.paparazziuav.org/wiki/ABI), necessary
 * any time data calculated in another module needs to be accessed. Including the file where this external
 * data is defined is not enough, since modules are executed parallel to each other, at different frequencies,
 * in different threads. The ABI event is triggered every time new data is sent out, and as such the function
 * defined in this file does not need to be explicitly called, only bound in the init function
 */
#ifndef CUSTOM_AVOIDER_CUSTOM_DETECTION_ID
#define CUSTOM_AVOIDER_CUSTOM_DETECTION_ID ABI_BROADCAST
#endif
static abi_event color_detection_ev;
static void object_detection_cb(uint8_t __attribute__((unused)) sender_id,
                               uint8_t __attribute__((unused)) left,
                               uint8_t __attribute__((unused)) middle, 
                               uint8_t __attribute__((unused)) right
                               )
{
  objectLeft = left;
  objectMiddle = middle;
  objectRight = right;
  updateWaypoint = true;
}

/*
 * Initialisation function, setting the colour filter, random seed and heading_increment
 */
void navigation_controller_init(void)
{
  uint32_t now_ms = get_sys_time_msec();

  // Initialise random values
  srand(time(NULL));
  chooseRandomIncrementAvoidance();

  // Allow first validated transitions to happen immediately.
  objectLeftLastSwitchMs = now_ms - CUSTOM_AVOIDER_HOLD_TIME_MS;
  objectMiddleLastSwitchMs = now_ms - CUSTOM_AVOIDER_HOLD_TIME_MS;
  objectRightLastSwitchMs = now_ms - CUSTOM_AVOIDER_HOLD_TIME_MS;

  // bind our colorfilter callbacks to receive the color filter outputs
  AbiBindMsgCUSTOM_DETECTION(CUSTOM_AVOIDER_CUSTOM_DETECTION_ID, &color_detection_ev, object_detection_cb);
}

/*
 * Function that checks it is safe to move forwards, and then moves a waypoint forward or changes the heading
 */
void navigation_controller_periodic(void)
{
  // only evaluate our state machine if we are flying
  if(!autopilot_in_flight()){
    return;
  }

  if(updateWaypoint){
    updateWaypoint = false;
  } else {
    return;
  }

  // bound obstacle_free_confidence

  float moveDistance = 0.5f; // default move distance [m]
  float headingIncrement = 1.0f; // default heading increment [deg]
  uint32_t now_ms = get_sys_time_msec();
 
  update_filtered_detection(objectLeft, &objectLeftFiltered, &objectLeftCandidate,
                            &objectLeftConfirmCount, &objectLeftLastSwitchMs, now_ms);
  update_filtered_detection(objectMiddle, &objectMiddleFiltered, &objectMiddleCandidate,
                            &objectMiddleConfirmCount, &objectMiddleLastSwitchMs, now_ms);
  update_filtered_detection(objectRight, &objectRightFiltered, &objectRightCandidate,
                            &objectRightConfirmCount, &objectRightLastSwitchMs, now_ms);

  VERBOSE_PRINT("Object detections - raw L:%d M:%d R:%d | filtered L:%d M:%d R:%d\n",
                objectLeft, objectMiddle, objectRight,
                objectLeftFiltered, objectMiddleFiltered, objectRightFiltered);


  switch (navigation_state){
    case SAFE:
      // Move waypoint forward
      moveWaypointForward(WP_TRAJECTORY, moveDistance);
      if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY),WaypointY(WP_TRAJECTORY))){
        navigation_state = OUT_OF_BOUNDS;
        //VERBOSE_PRINT("Sate: OUT_OF_BOUNDS\n");
      } else if(objectMiddleFiltered == 1){
        navigation_state = OBSTACLE_MIDDLE;
        //VERBOSE_PRINT("State: OBSTACLE_MIDDLE\n");
      } else if (objectLeftFiltered == 1 && objectRightFiltered == 0){
        navigation_state = OBSTACLE_LEFT;
        //VERBOSE_PRINT("State: OBSTACLE_LEFT\n");
      } else if (objectRightFiltered == 1 && objectLeftFiltered == 0){
        navigation_state = OBSTACLE_RIGHT;
        //VERBOSE_PRINT("State: OBSTACLE_RIGHT\n");
      } else {
            moveWaypointForward(WP_GOAL, moveDistance);
      }

      break;
    case OBSTACLE_LEFT:
      // stop
      waypoint_move_here_2d(WP_GOAL);
      waypoint_move_here_2d(WP_TRAJECTORY);

      increase_nav_heading(headingIncrement);

      if(objectLeftFiltered == 0){
        navigation_state = SAFE;
        //VERBOSE_PRINT("State: SAFE\n");
      }
 
      break;

    case OBSTACLE_RIGHT:
      // stop
      waypoint_move_here_2d(WP_GOAL);
      waypoint_move_here_2d(WP_TRAJECTORY);

      increase_nav_heading(-headingIncrement);

      if(objectRightFiltered == 0){
        navigation_state = SAFE;
        //VERBOSE_PRINT("State: SAFE\n");
      }
    
      break;

    case OBSTACLE_MIDDLE:
      // stop
      waypoint_move_here_2d(WP_GOAL);
      waypoint_move_here_2d(WP_TRAJECTORY);

      chooseRandomIncrementAvoidance();
      navigation_state = SEARCH_FOR_SAFE_HEADING;
      //VERBOSE_PRINT("State: SEARCH_FOR_SAFE_HEADING\n");

      break;

    case SEARCH_FOR_SAFE_HEADING:
    increase_nav_heading(heading_increment);

      // make sure we have a couple of good readings before declaring the way safe
      if (objectLeftFiltered == 0 && objectMiddleFiltered == 0 && objectRightFiltered == 0){
        navigation_state = SAFE;
        //VERBOSE_PRINT("State: SAFE\n");
      }
      break;
    case OUT_OF_BOUNDS:
      increase_nav_heading(heading_increment);
      moveWaypointForward(WP_TRAJECTORY, 1.5f);

      if (InsideObstacleZone(WaypointX(WP_TRAJECTORY),WaypointY(WP_TRAJECTORY))){
        // add offset to head back into arena
        increase_nav_heading(heading_increment);

        // ensure direction is safe before continuing
        navigation_state = SEARCH_FOR_SAFE_HEADING;
        //VERBOSE_PRINT("State: SEARCH_FOR_SAFE_HEADING\n");
      }
      break;
    default:
      break;
  }
  return;
}

/*
 * Filter binary detections using K-confirm transitions plus a minimum switch hold time.
 */
static void update_filtered_detection(uint8_t raw_detection, uint8_t *filtered_detection,
                                      uint8_t *candidate_detection, uint8_t *confirm_count,
                                      uint32_t *last_switch_ms, uint32_t now_ms)
{
  if (raw_detection == *filtered_detection) {
    *confirm_count = 0;
    *candidate_detection = raw_detection;
    return;
  }

  if (raw_detection != *candidate_detection) {
    *candidate_detection = raw_detection;
    *confirm_count = 1;
  } else if (*confirm_count < 255) {
    (*confirm_count)++;
  }

  if ((*confirm_count >= CUSTOM_AVOIDER_CONFIRM_SAMPLES) &&
      ((now_ms - *last_switch_ms) >= CUSTOM_AVOIDER_HOLD_TIME_MS)) {
    *filtered_detection = raw_detection;
    *last_switch_ms = now_ms;
    *confirm_count = 0;
  }
}


/*
 * Increases the NAV heading. Assumes heading is an INT32_ANGLE. It is bound in this function.
 */
uint8_t increase_nav_heading(float incrementDegrees)
{
  float new_heading = stateGetNedToBodyEulers_f()->psi + RadOfDeg(incrementDegrees);

  // normalize heading to [-pi, pi]
  FLOAT_ANGLE_NORMALIZE(new_heading);

  // set heading, declared in firmwares/rotorcraft/navigation.h
  nav.heading = new_heading;

  //VERBOSE_PRINT("Increasing heading to %f\n", DegOfRad(new_heading));
  return false;
}

/*
 * Calculates coordinates of distance forward and sets waypoint 'waypoint' to those coordinates
 */
uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters)
{
  struct EnuCoor_i new_coor;
  calculateForwards(&new_coor, distanceMeters);
  moveWaypoint(waypoint, &new_coor);
  return false;
}

/*
 * Calculates coordinates of a distance of 'distanceMeters' forward w.r.t. current position and heading
 */
uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters)
{
  float heading  = stateGetNedToBodyEulers_f()->psi;

  // Now determine where to place the waypoint you want to go to
  new_coor->x = stateGetPositionEnu_i()->x + POS_BFP_OF_REAL(sinf(heading) * (distanceMeters));
  new_coor->y = stateGetPositionEnu_i()->y + POS_BFP_OF_REAL(cosf(heading) * (distanceMeters));
  // VERBOSE_PRINT("Calculated %f m forward position. x: %f  y: %f based on pos(%f, %f) and heading(%f)\n", distanceMeters,	
  //               POS_FLOAT_OF_BFP(new_coor->x), POS_FLOAT_OF_BFP(new_coor->y),
  //               stateGetPositionEnu_f()->x, stateGetPositionEnu_f()->y, DegOfRad(heading));
  return false;
}

/*
 * Sets waypoint 'waypoint' to the coordinates of 'new_coor'
 */
uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor)
{
  // VERBOSE_PRINT("Moving waypoint %d to x:%f y:%f\n", waypoint, POS_FLOAT_OF_BFP(new_coor->x),
  //               POS_FLOAT_OF_BFP(new_coor->y));
  waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);
  return false;
}

/*
 * Sets the variable 'heading_increment' randomly positive/negative
 */
uint8_t chooseRandomIncrementAvoidance(void)
{
  // Randomly choose CW or CCW avoiding direction
  if (rand() % 2 == 0) {
    heading_increment = 5.f;
    //VERBOSE_PRINT("Set avoidance increment to: %f\n", heading_increment);
  } else {
    heading_increment = -5.f;
    //VERBOSE_PRINT("Set avoidance increment to: %f\n", heading_increment);
  }
  return false;
}

