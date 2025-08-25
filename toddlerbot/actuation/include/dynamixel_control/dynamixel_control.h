#pragma once

#include "dynamixel_client.h"
#include <vector>
#include <string>
#include <mutex>
#include <tuple>
#include <memory>

/// Higher-level wrapper around DynamixelClient, mirroring the Python DynamixelController API.
class DynamixelControl
{
public:
  /// Construct with port name, motor IDs, PID gains, etc.
  DynamixelControl(const std::string &port,
                   const std::vector<int> &motor_ids,
                   const std::vector<float> &kp,
                   const std::vector<float> &kd,
                   const std::vector<float> &zero_pos = {},
                   const std::vector<std::string> &control_mode = {},
                   int baudrate = 2000000,
                   int return_delay_time = 1);

  ~DynamixelControl();

  /// Set proportional gain(s).
  void set_kp(const std::vector<float> &kp);

  /// Set both proportional and derivative gain.
  void set_kp_kd(float kp, float kd);

  /// Bulk-set any of kp/kd/ki/kff1/kff2 on a subset of motor IDs.
  void set_parameters(const std::vector<int> &ids,
                      float kp, float kd, float ki,
                      float kff1, float kff2);

  /// Write desired current.
  void set_current(const std::vector<float> &cur);

  /// Perform reboot, voltage check, and register initialization.
  void initialize_motors();

  /// Force-close all open clients.
  void close_motors();

  /// Disable torque on given IDs (or all if empty).
  void disable_motors(const std::vector<int> &ids = {});

  /// Enable torque on given IDs (or all if empty).
  void enable_motors(const std::vector<int> &ids = {});

  /// Write a new position command (in radians) to the motors.
  void set_pos(const std::vector<float> &pos);

  /// Read back (time, pos, vel, cur).
  std::map<std::string, std::vector<float>>
  get_state(int retries = 0);

  const std::vector<int> &get_motor_ids() const { return motor_ids_; }

private:
  std::string port_;
  std::vector<int> motor_ids_;
  std::vector<float> kp_, kd_, ki_, kff1_, kff2_;
  std::vector<float> zero_pos_;
  std::vector<std::string> control_mode_;
  int baudrate_;
  int return_delay_time_;

  std::unique_ptr<DynamixelClient> client_;
};
