#pragma once

#include <Arduino.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "bram_controller_data.hpp"

namespace bram {

struct ServoAction {
  float values[3];
};

struct ArcCommandView {
  const bram_data::ArcGridPoint* grid;
  std::size_t gridCount;
  bram_data::ArcParams fallback;
};

class BramController {
 public:
  ServoAction action(float forwardCommand,
                     float yawCommand,
                     std::uint32_t step,
                     float headingError = 0.0f,
                     float yawRate = 0.0f) const {
    const float forward = clip(forwardCommand, -1.0f, 1.0f);
    const float yaw = clip(yawCommand, -1.0f, 1.0f);
    const float forwardMag = std::fabs(forward);
    const float yawMag = std::fabs(yaw);
    const float t = static_cast<float>(step) * bram_data::kDt;

    if (forwardMag < 0.05f && yawMag < 0.05f) {
      return ServoAction{};
    }
    if (yawMag < 0.05f) {
      if (translationTableRows(forward) > 0) {
        return translationAction(forward, step);
      }
      float base[3];
      baseAction(forward, yaw, t, headingError, yawRate, base);
      return servoActionFromArray(base);
    }
    if (forwardMag < 0.05f) {
      float yawValues[3];
      yawAction(yaw, step, yawValues);
      return servoActionFromArray(yawValues);
    }

    float base[3];
    baseAction(forward, yaw, t, headingError, yawRate, base);

    const float gate = residualGate(forward, yaw);
    if (gate <= 1.0e-6f) {
      return teacherAction(forward, yaw, step, base);
    }

    const ArcCommandView view = commandView(forward, yaw);
    const bram_data::ArcParams params =
        selectArcParams(view, std::fabs(forward), std::fabs(yaw));

    float yawActionValues[3];
    yawAction(yaw, static_cast<int64_t>(step) + params.stepOffset, yawActionValues);

    ServoAction out{};
    for (std::size_t i = 0; i < 3; ++i) {
      const float baseDelta = (params.baseScale - 1.0f) * base[i];
      const float rawResidual =
          (baseDelta + params.yawScales[i] * yawActionValues[i]) /
          maxFloat(1.0e-6f, bram_data::kResidualLimit * gate);
      const float residual = clip(rawResidual, -1.0f, 1.0f);
      out.values[i] = clip(base[i] + bram_data::kResidualLimit * gate * residual,
                           -1.0f,
                           1.0f);
    }
    return out;
  }

 private:
  static constexpr float kHeadingTrimLimit = 0.35f;

  static float clip(float value, float low, float high) {
    if (value < low) return low;
    if (value > high) return high;
    return value;
  }

  static float maxFloat(float a, float b) { return a > b ? a : b; }

  static float smoothstep(float x) {
    x = clip(x, 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
  }

  static float residualGate(float forward, float yaw) {
    return smoothstep(std::fabs(forward) / 0.28f) *
           smoothstep(std::fabs(yaw) / 0.22f);
  }

  static void scaledParams(float forwardCommand, float yawCommand, float params[21]) {
    const float forward = forwardCommand;
    const float yaw = yawCommand;
    const float forwardMag = std::fabs(forward);
    const float yawMag = std::fabs(yaw);
    const float activity = clip(maxFloat(forwardMag, yawMag), 0.0f, 1.0f);
    const float* source = forward >= 0.0f ? bram_data::kForwardParams
                                          : bram_data::kBackwardParams;
    if (forwardMag < 0.05f && yawMag >= 0.05f) {
      source = bram_data::kForwardParams;
    }

    const float speedScale =
        activity < 0.05f
            ? 1.0f
            : bram_data::kBaseSpeedMin + (1.0f - bram_data::kBaseSpeedMin) * activity;
    const float actionScale =
        activity < 0.05f
            ? 0.0f
            : bram_data::kBaseActionMin + (1.0f - bram_data::kBaseActionMin) * activity;

    for (std::size_t i = 0; i < 21; ++i) {
      params[i] = source[i];
    }
    params[0] = source[0] * speedScale;
    for (std::size_t i = 1; i < 4; ++i) params[i] = source[i] * actionScale;
    for (std::size_t i = 4; i < 7; ++i) params[i] = source[i] * actionScale;
    for (std::size_t i = 10; i < 13; ++i) params[i] = source[i] * actionScale;
  }

  static void gaitAction(const float params[21],
                         float t,
                         float headingError,
                         float yawRate,
                         bool useHeadingCorrection,
                         float out[3]) {
    const float thetaBase = 2.0f * PI * params[0] * t;
    for (std::size_t i = 0; i < 3; ++i) {
      const float theta = thetaBase + params[7 + i];
      out[i] = params[1 + i] + params[4 + i] * std::sin(theta) +
               params[10 + i] * std::sin(2.0f * theta + params[18 + i]);
    }

    if (useHeadingCorrection) {
      float trim = -params[13] * headingError - params[14] * yawRate;
      trim = clip(trim, -kHeadingTrimLimit, kHeadingTrimLimit);
      for (std::size_t i = 0; i < 3; ++i) {
        out[i] += trim * params[15 + i];
      }
    }

    for (std::size_t i = 0; i < 3; ++i) {
      out[i] = clip(out[i], -1.0f, 1.0f);
    }
  }

  static void baseAction(float forwardCommand,
                         float yawCommand,
                         float t,
                         float headingError,
                         float yawRate,
                         float out[3]) {
    float params[21];
    scaledParams(forwardCommand, yawCommand, params);
    const bool useHeadingCorrection =
        std::fabs(forwardCommand) >= 0.05f && std::fabs(yawCommand) < 0.05f;
    gaitAction(params, t, headingError, yawRate, useHeadingCorrection, out);
  }

  static int positiveMod(int64_t value, int mod) {
    int result = static_cast<int>(value % mod);
    return result < 0 ? result + mod : result;
  }

  static ServoAction servoActionFromArray(const float values[3]) {
    ServoAction out{};
    for (std::size_t i = 0; i < 3; ++i) {
      out.values[i] = clip(values[i], -1.0f, 1.0f);
    }
    return out;
  }

  static ServoAction translationAction(float forwardCommand, std::uint32_t step) {
    const float magnitude = std::fabs(forwardCommand);
    ServoAction out{};
    if (magnitude < 1.0e-6f) {
      return out;
    }
    const bool forward = forwardCommand > 0.0f;
    const std::size_t rows =
        forward ? bram_data::kForwardTableRows : bram_data::kBackwardTableRows;
    const int row = positiveMod(step, static_cast<int>(rows));
    for (std::size_t i = 0; i < 3; ++i) {
      const float value = forward ? bram_data::kForwardTable[row][i]
                                  : bram_data::kBackwardTable[row][i];
      out.values[i] = clip(magnitude * value, -1.0f, 1.0f);
    }
    return out;
  }

  static std::size_t translationTableRows(float forwardCommand) {
    return forwardCommand > 0.0f ? bram_data::kForwardTableRows
                                 : bram_data::kBackwardTableRows;
  }

  static void yawAction(float yawCommand, int64_t step, float out[3]) {
    const float magnitude = std::fabs(yawCommand);
    if (magnitude < 1.0e-6f) {
      out[0] = out[1] = out[2] = 0.0f;
      return;
    }

    const bool left = yawCommand > 0.0f;
    const std::size_t rows =
        left ? bram_data::kYawLeftTableRows : bram_data::kYawRightTableRows;
    const int row = positiveMod(step, static_cast<int>(rows));
    for (std::size_t i = 0; i < 3; ++i) {
      const float value = left ? bram_data::kYawLeftTable[row][i]
                               : bram_data::kYawRightTable[row][i];
      out[i] = clip(magnitude * value, -1.0f, 1.0f);
    }
  }

  static ServoAction teacherAction(float forward,
                                   float yaw,
                                   std::uint32_t step,
                                   const float base[3]) {
    const float forwardMag = std::fabs(forward);
    const float yawMag = std::fabs(yaw);
    ServoAction out{};
    if (forwardMag < 0.05f && yawMag < 0.05f) {
      return out;
    }
    if (yawMag < 0.05f) {
      for (std::size_t i = 0; i < 3; ++i) out.values[i] = base[i];
      return out;
    }

    float yawValues[3];
    yawAction(yaw, step, yawValues);
    if (forwardMag < 0.05f) {
      for (std::size_t i = 0; i < 3; ++i) out.values[i] = yawValues[i];
      return out;
    }

    for (std::size_t i = 0; i < 3; ++i) {
      out.values[i] = clip(base[i] + bram_data::kArcYawScale * yawValues[i],
                           -1.0f,
                           1.0f);
    }
    return out;
  }

  static ArcCommandView commandView(float forward, float yaw) {
    if (forward >= 0.0f && yaw >= 0.0f) {
      return {bram_data::kArcFlGrid, bram_data::kArcFlGridCount,
              bram_data::kArcFlDefault};
    }
    if (forward >= 0.0f && yaw < 0.0f) {
      return {bram_data::kArcFrGrid, bram_data::kArcFrGridCount,
              bram_data::kArcFrDefault};
    }
    if (forward < 0.0f && yaw >= 0.0f) {
      return {bram_data::kArcBlGrid, bram_data::kArcBlGridCount,
              bram_data::kArcBlDefault};
    }
    return {bram_data::kArcBrGrid, bram_data::kArcBrGridCount,
            bram_data::kArcBrDefault};
  }

  static void bracketAxis(const ArcCommandView& view,
                          float target,
                          bool forwardAxis,
                          float& low,
                          float& high,
                          float& t) {
    float minValue = forwardAxis ? view.grid[0].forward : view.grid[0].yaw;
    float maxValue = minValue;
    for (std::size_t i = 1; i < view.gridCount; ++i) {
      const float value = forwardAxis ? view.grid[i].forward : view.grid[i].yaw;
      if (value < minValue) minValue = value;
      if (value > maxValue) maxValue = value;
    }
    if (target <= minValue) {
      low = high = minValue;
      t = 0.0f;
      return;
    }
    if (target >= maxValue) {
      low = high = maxValue;
      t = 0.0f;
      return;
    }

    low = minValue;
    high = maxValue;
    for (std::size_t i = 0; i < view.gridCount; ++i) {
      const float value = forwardAxis ? view.grid[i].forward : view.grid[i].yaw;
      if (value <= target && value > low) low = value;
      if (value >= target && value < high) high = value;
    }
    t = high == low ? 0.0f : (target - low) / (high - low);
  }

  static bram_data::ArcParams gridParams(const ArcCommandView& view,
                                         float gridForward,
                                         float gridYaw,
                                         float targetForward,
                                         float targetYaw) {
    for (std::size_t i = 0; i < view.gridCount; ++i) {
      if (std::fabs(view.grid[i].forward - gridForward) < 0.0001f &&
          std::fabs(view.grid[i].yaw - gridYaw) < 0.0001f) {
        return view.grid[i].params;
      }
    }

    std::size_t best = 0;
    float bestDistance = 1.0e9f;
    for (std::size_t i = 0; i < view.gridCount; ++i) {
      const float df = view.grid[i].forward - targetForward;
      const float dy = view.grid[i].yaw - targetYaw;
      const float distance = df * df + dy * dy;
      if (distance < bestDistance) {
        bestDistance = distance;
        best = i;
      }
    }
    return view.grid[best].params;
  }

  static bram_data::ArcParams blendArcParams(const bram_data::ArcParams& p00,
                                             const bram_data::ArcParams& p10,
                                             const bram_data::ArcParams& p01,
                                             const bram_data::ArcParams& p11,
                                             float ft,
                                             float yt) {
    const float weights[4] = {
        (1.0f - ft) * (1.0f - yt),
        ft * (1.0f - yt),
        (1.0f - ft) * yt,
        ft * yt,
    };
    const bram_data::ArcParams params[4] = {p00, p10, p01, p11};
    bram_data::ArcParams out{};
    float stepOffset = 0.0f;
    for (std::size_t i = 0; i < 4; ++i) {
      out.baseScale += weights[i] * params[i].baseScale;
      stepOffset += weights[i] * static_cast<float>(params[i].stepOffset);
      for (std::size_t servo = 0; servo < 3; ++servo) {
        out.yawScales[servo] += weights[i] * params[i].yawScales[servo];
      }
    }
    out.stepOffset = static_cast<int>(std::round(stepOffset));
    return out;
  }

  static bram_data::ArcParams selectArcParams(const ArcCommandView& view,
                                              float forwardMag,
                                              float yawMag) {
    if (view.gridCount == 0) {
      return view.fallback;
    }
    float f0;
    float f1;
    float ft;
    float y0;
    float y1;
    float yt;
    bracketAxis(view, forwardMag, true, f0, f1, ft);
    bracketAxis(view, yawMag, false, y0, y1, yt);

    const bram_data::ArcParams p00 = gridParams(view, f0, y0, forwardMag, yawMag);
    const bram_data::ArcParams p10 = gridParams(view, f1, y0, forwardMag, yawMag);
    const bram_data::ArcParams p01 = gridParams(view, f0, y1, forwardMag, yawMag);
    const bram_data::ArcParams p11 = gridParams(view, f1, y1, forwardMag, yawMag);
    return blendArcParams(p00, p10, p01, p11, ft, yt);
  }
};

}  // namespace bram
