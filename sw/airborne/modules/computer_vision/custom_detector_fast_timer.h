#ifndef CUSTOM_DETECTOR_FAST_H
#define CUSTOM_DETECTOR_FAST_H

#include "std.h"
#include "modules/computer_vision/cv.h"

#ifdef __cplusplus
extern "C" {
#endif

void custom_color_risk_detector_init(void);
void custom_color_risk_detector_periodic(void);

#ifdef __cplusplus
}
#endif

#endif /* CV_CUSTOM_COLOR_RISK_DETECTOR_H */