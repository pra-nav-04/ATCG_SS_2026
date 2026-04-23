#include "opg/gui/statsdisplay.h"

#include <imgui/imgui.h>

namespace opg {

static void displayText(const char* text, float x, float y)
{
    ImVec2 viewport_pos = ImGui::GetWindowViewport()->Pos;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::SetNextWindowBgAlpha( 0.0f );
    ImGui::SetNextWindowPos( ImVec2( viewport_pos.x + x, viewport_pos.y + y ) );
    ImGui::Begin( "TextOverlayFG", nullptr,
                  ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize
                      | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoInputs );
    ImGui::TextColored( ImColor( 0.7f, 0.7f, 0.7f, 1.0f ), "%s", text );
    ImGui::End();
    ImGui::PopStyleVar();
}

StatsDisplay::StatsDisplay(Window *_window, double _alpha) :
    window { _window },
    alpha { _alpha }
{
}

void StatsDisplay::setVisibility(bool _visible)
{
    visible = _visible;
}

StatsDisplay::duration StatsDisplay::update_avg_time(duration& avg_time, duration current_time)
{
    avg_time = (1-alpha) * avg_time + alpha * current_time;
    return avg_time;
}

void StatsDisplay::updateStats(duration state_update_time,
                                duration render_time,
                                duration display_time)
{
    const auto current_time = clock::now();

    auto frame_time = std::chrono::duration_cast<duration>(current_time - last_time);
    last_time = current_time;

    update_avg_time(avg_frame_time, frame_time);
    update_avg_time(avg_state_update_time, state_update_time);
    update_avg_time(avg_render_time, render_time);
    update_avg_time(avg_display_time, display_time);

    if (current_time - last_update_time > min_time_between_updates)
    {
        display_text.resize(256, '\0');

        typedef std::chrono::duration<double, std::milli> millisecond_duration;

        millisecond_duration avg_frame_time_ms        = avg_frame_time;
        millisecond_duration avg_state_update_time_ms = avg_state_update_time;
        millisecond_duration avg_render_time_ms       = avg_render_time;
        millisecond_duration avg_display_time_ms      = avg_display_time;

        auto avg_fps = 1 / avg_frame_time.count();

        sprintf( display_text.data(),
                 "frame       : %8.1f ms | %5.1f fps\n\n"
                 "state update: %8.1f ms\n"
                 "render      : %8.1f ms\n"
                 "display     : %8.1f ms\n",
                 avg_frame_time_ms.count(), avg_fps,
                 avg_state_update_time_ms.count(),
                 avg_render_time_ms.count(),
                 avg_display_time_ms.count() );

        last_update_time = current_time;
    }
}

void StatsDisplay::renderGui()
{
    if (visible)
    {
        displayText(display_text.c_str(), 10.0f, 10.0f);
    }
}

} // end namespace opg
