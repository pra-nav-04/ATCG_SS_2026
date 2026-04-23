#pragma once

#include <string>
#include <chrono>

#include "opg/gui/window.h"
#include "opg/opgapi.h"

namespace opg {

class StatsDisplay
{
public:
    OPG_API StatsDisplay(Window *window, double alpha = 0.9);

    typedef std::chrono::steady_clock clock;
    typedef std::chrono::duration<double> duration;

    OPG_API void updateStats(duration state_update_time,
                      duration render_time,
                      duration display_time);

    inline bool getVisibility() const { return visible; }
    inline void toggleVisibility() { visible = !visible; }
    OPG_API void setVisibility(bool visible);

    OPG_API void renderGui();

private:
    OPG_API duration update_avg_time(duration& avg_time, duration current_time);

private:
    Window*             window;

    bool                visible                     = true;

    double              alpha; // falloff coefficient, 1 => average is current frame

    clock::time_point   last_time                   = clock::now();

    // Last update of display text
    clock::time_point   last_update_time            = clock::time_point();
    clock::duration     min_time_between_updates    = std::chrono::duration_cast<clock::duration>(duration(0.5));

    duration            avg_frame_time              = duration::zero();
    duration            avg_state_update_time       = duration::zero();
    duration            avg_render_time             = duration::zero();
    duration            avg_display_time            = duration::zero();

    std::string         display_text;
};

} // end namespace opg
