#pragma once

#include <Arduino.h>
#include <cmath>
#include <cstddef>

#include "bram_controller.hpp"
#include "bram_policy_data.hpp"

namespace bram {

class BramPolicyController {
 public:
  static constexpr std::size_t kYawPolicyHoldTicks = 2;

  BramPolicyController() {
    const float identityQuat[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    reset(identityQuat);
  }

  void reset(const float quatWxyz[4]) {
    float normalized[4];
    normalizeQuat(quatWxyz, normalized);
    fillHistories(normalized);
    activePrimitive_ = Primitive::Idle;
    yawHoldTicksRemaining_ = 0;
    cachedAction_ = ServoAction{};
  }

  ServoAction action(float forwardCommand, float yawCommand, const float quatWxyz[4]) {
    const float forward = clip(forwardCommand, -1.0f, 1.0f);
    const float yaw = clip(yawCommand, -1.0f, 1.0f);
    const Primitive primitive = selectPrimitive(forward, yaw);

    float quat[4];
    normalizeQuat(quatWxyz, quat);

    if (primitive == Primitive::Idle) {
      reset(quat);
      return ServoAction{};
    }

    if (primitive != activePrimitive_) {
      fillHistories(quat);
      activePrimitive_ = primitive;
      yawHoldTicksRemaining_ = 0;
    } else if (primitive == Primitive::Yaw && yawHoldTicksRemaining_ > 0) {
      --yawHoldTicksRemaining_;
      return cachedAction_;
    } else {
      pushImu(quat);
    }

    const float commandScalar = primitive == Primitive::Yaw ? yaw : 0.0f;
    float obs[bram_policy_data::kObsDim];
    buildObservation(commandScalar, obs);

    float output[bram_policy_data::kActionDim];
    infer(weightsFor(primitive), obs, output);

    const float scale =
        primitive == Primitive::Forward || primitive == Primitive::Backward
            ? std::fabs(forward)
            : 1.0f;

    ServoAction out{};
    for (std::size_t i = 0; i < bram_policy_data::kServoCount; ++i) {
      out.values[i] = clip(scale * output[i], -1.0f, 1.0f);
    }
    pushAction(out.values);
    cachedAction_ = out;
    yawHoldTicksRemaining_ =
        primitive == Primitive::Yaw ? kYawPolicyHoldTicks - 1 : 0;
    return out;
  }

 private:
  enum class Primitive {
    Idle,
    Forward,
    Backward,
    Yaw,
  };

  Primitive activePrimitive_ = Primitive::Idle;
  std::size_t yawHoldTicksRemaining_ = 0;
  ServoAction cachedAction_{};
  float imuHistory_[bram_policy_data::kImuHistoryFrames][bram_policy_data::kImuFrameDim]{};
  float actionHistory_[bram_policy_data::kActionHistoryFrames][bram_policy_data::kServoCount]{};

  static float clip(float value, float low, float high) {
    if (value < low) return low;
    if (value > high) return high;
    return value;
  }

  static Primitive selectPrimitive(float forward, float yaw) {
    if (std::fabs(forward) >= 0.05f) {
      return forward > 0.0f ? Primitive::Forward : Primitive::Backward;
    }
    if (std::fabs(yaw) >= 0.05f) {
      return Primitive::Yaw;
    }
    return Primitive::Idle;
  }

  static const bram_policy_data::ActorWeights& weightsFor(Primitive primitive) {
    switch (primitive) {
      case Primitive::Forward:
        return bram_policy_data::kForwardPolicy;
      case Primitive::Backward:
        return bram_policy_data::kBackwardPolicy;
      case Primitive::Yaw:
        return bram_policy_data::kYawPolicy;
      case Primitive::Idle:
      default:
        return bram_policy_data::kForwardPolicy;
    }
  }

  static void normalizeQuat(const float in[4], float out[4]) {
    float norm = std::sqrt(
        in[0] * in[0] + in[1] * in[1] + in[2] * in[2] + in[3] * in[3]);
    if (!(norm > 1.0e-6f)) {
      norm = 1.0f;
      out[0] = 1.0f;
      out[1] = out[2] = out[3] = 0.0f;
      return;
    }
    const float invNorm = 1.0f / norm;
    for (std::size_t i = 0; i < 4; ++i) {
      out[i] = in[i] * invNorm;
    }
  }

  void fillHistories(const float quat[4]) {
    for (std::size_t frame = 0; frame < bram_policy_data::kImuHistoryFrames; ++frame) {
      for (std::size_t axis = 0; axis < bram_policy_data::kImuFrameDim; ++axis) {
        imuHistory_[frame][axis] = quat[axis];
      }
    }
    for (std::size_t frame = 0; frame < bram_policy_data::kActionHistoryFrames; ++frame) {
      for (std::size_t servo = 0; servo < bram_policy_data::kServoCount; ++servo) {
        actionHistory_[frame][servo] = 0.0f;
      }
    }
  }

  void pushImu(const float quat[4]) {
    for (std::size_t frame = 0; frame + 1 < bram_policy_data::kImuHistoryFrames; ++frame) {
      for (std::size_t axis = 0; axis < bram_policy_data::kImuFrameDim; ++axis) {
        imuHistory_[frame][axis] = imuHistory_[frame + 1][axis];
      }
    }
    for (std::size_t axis = 0; axis < bram_policy_data::kImuFrameDim; ++axis) {
      imuHistory_[bram_policy_data::kImuHistoryFrames - 1][axis] = quat[axis];
    }
  }

  void pushAction(const float action[3]) {
    for (std::size_t frame = 0; frame + 1 < bram_policy_data::kActionHistoryFrames; ++frame) {
      for (std::size_t servo = 0; servo < bram_policy_data::kServoCount; ++servo) {
        actionHistory_[frame][servo] = actionHistory_[frame + 1][servo];
      }
    }
    for (std::size_t servo = 0; servo < bram_policy_data::kServoCount; ++servo) {
      actionHistory_[bram_policy_data::kActionHistoryFrames - 1][servo] =
          clip(action[servo], -1.0f, 1.0f);
    }
  }

  void buildObservation(float commandScalar, float obs[bram_policy_data::kObsDim]) const {
    std::size_t index = 0;
    for (std::size_t frame = 0; frame < bram_policy_data::kImuHistoryFrames; ++frame) {
      for (std::size_t axis = 0; axis < bram_policy_data::kImuFrameDim; ++axis) {
        obs[index++] = imuHistory_[frame][axis];
      }
    }
    for (std::size_t frame = 0; frame < bram_policy_data::kActionHistoryFrames; ++frame) {
      for (std::size_t servo = 0; servo < bram_policy_data::kServoCount; ++servo) {
        obs[index++] = actionHistory_[frame][servo];
      }
    }
    obs[index++] = clip(commandScalar, -1.0f, 1.0f);
  }

  static void infer(const bram_policy_data::ActorWeights& weights,
                    const float obs[bram_policy_data::kObsDim],
                    float out[bram_policy_data::kActionDim]) {
    float h0[bram_policy_data::kHiddenDim];
    float h1[bram_policy_data::kHiddenDim];

    for (std::size_t row = 0; row < bram_policy_data::kHiddenDim; ++row) {
      float sum = weights.b0[row];
      for (std::size_t col = 0; col < bram_policy_data::kObsDim; ++col) {
        sum += weights.w0[row][col] * obs[col];
      }
      h0[row] = std::tanh(sum);
    }

    for (std::size_t row = 0; row < bram_policy_data::kHiddenDim; ++row) {
      float sum = weights.b1[row];
      for (std::size_t col = 0; col < bram_policy_data::kHiddenDim; ++col) {
        sum += weights.w1[row][col] * h0[col];
      }
      h1[row] = std::tanh(sum);
    }

    for (std::size_t row = 0; row < bram_policy_data::kActionDim; ++row) {
      float sum = weights.b2[row];
      for (std::size_t col = 0; col < bram_policy_data::kHiddenDim; ++col) {
        sum += weights.w2[row][col] * h1[col];
      }
      out[row] = clip(std::tanh(sum), -1.0f, 1.0f);
    }
  }
};

}  // namespace bram
