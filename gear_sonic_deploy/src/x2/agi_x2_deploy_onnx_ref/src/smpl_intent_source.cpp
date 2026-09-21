#include "zmq/smpl_intent_source.hpp"

#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace agi_x2 {

namespace {
double SteadyNowS()
{
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool CopyF32(const ZMQPackedMessageSubscriber::FieldInfo& f,
             const ZMQPackedMessageSubscriber::BufferView& b,
             float* out, std::size_t n)
{
  if (f.dtype != "f32" || b.size < n * sizeof(float)) return false;
  std::memcpy(out, b.data, n * sizeof(float));
  return true;
}
}  // namespace

std::unique_ptr<SmplIntentSource> SmplIntentSource::Open(
    const std::string& host, int port, bool bind, const std::string& topic)
{
  std::unique_ptr<SmplIntentSource> src(new SmplIntentSource());
  src->sub_ = std::make_unique<ZMQPackedMessageSubscriber>(
      host, port, topic, /*timeout_ms=*/200, /*verbose=*/false,
      /*conflate=*/false, /*rcv_hwm=*/64, bind);
  src->sub_->SetOnDecodedMessage(
      [self = src.get()](const std::string& t,
                         const ZMQPackedMessageSubscriber::DecodedHeader& h,
                         const std::vector<ZMQPackedMessageSubscriber::BufferView>& b) {
        self->HandleDecoded(t, h, b);
      });
  if (!src->sub_->Connect()) {
    throw std::runtime_error("SmplIntentSource: failed to " +
                             std::string(bind ? "bind" : "connect") + " tcp://" +
                             host + ":" + std::to_string(port));
  }
  src->sub_->Start();
  return src;
}

SmplIntentSource::~SmplIntentSource()
{
  if (sub_) sub_->Stop();
}

void SmplIntentSource::HandleDecoded(
    const std::string& /*topic*/,
    const ZMQPackedMessageSubscriber::DecodedHeader& header,
    const std::vector<ZMQPackedMessageSubscriber::BufferView>& buffers)
{
  SmplFrame fr;
  bool got_joints = false, got_quat = false;
  double engaged = 0.0;
  std::array<double, 4> wrist{0, 0, 0, 0};
  bool got_wrist = false;
  std::array<double, 2> wyaw{0, 0};
  bool got_wyaw = false;
  for (std::size_t i = 0; i < header.fields.size() && i < buffers.size(); ++i) {
    const auto& f = header.fields[i];
    const auto& b = buffers[i];
    if (f.name == "smpl_joints") {
      got_joints = CopyF32(f, b, fr.joints.data(), 72);
    } else if (f.name == "human_quat") {
      float q[4];
      if (CopyF32(f, b, q, 4)) {
        for (int k = 0; k < 4; ++k) fr.quat_wxyz[k] = q[k];
        got_quat = true;
      }
    } else if (f.name == "engaged") {
      float e;
      if (CopyF32(f, b, &e, 1)) engaged = e;
    } else if (f.name == "wrist_pr") {
      float w[4];
      if (CopyF32(f, b, w, 4)) {
        for (int k = 0; k < 4; ++k) wrist[k] = w[k];
        got_wrist = true;
      }
    } else if (f.name == "wrist_yaw") {
      float y[2];
      if (CopyF32(f, b, y, 2)) {
        wyaw[0] = y[0]; wyaw[1] = y[1];
        got_wyaw = true;
      }
    }
  }
  if (!got_joints || !got_quat) return;   // not an intent frame
  fr.t_mono = SteadyNowS();
  std::lock_guard<std::mutex> lock(mu_);
  ring_.push_back(fr);
  while (ring_.size() > 128) ring_.pop_front();
  last_rx_ = fr.t_mono;
  engaged_ = engaged;
  if (got_wrist) wrist_pr_ = wrist;
  if (got_wyaw) { wrist_yaw_ = wyaw; has_wrist_yaw_ = true; }
  frames_.fetch_add(1);
}

double SmplIntentSource::AgeS(double now_mono) const
{
  std::lock_guard<std::mutex> lock(mu_);
  if (last_rx_ < 0.0) return std::numeric_limits<double>::infinity();
  return now_mono - last_rx_;
}

bool SmplIntentSource::Alive() const
{
  std::lock_guard<std::mutex> lock(mu_);
  if (ring_.size() < 25) return false;
  const SmplFrame& a = ring_.back();
  const SmplFrame& b = ring_[ring_.size() - 25];
  return !(a.joints == b.joints && a.quat_wxyz == b.quat_wxyz);
}

bool SmplIntentSource::Fresh(double now_mono, double stale_s) const
{
  return AgeS(now_mono) < stale_s && Alive();
}

double SmplIntentSource::LatestEngaged() const
{
  std::lock_guard<std::mutex> lock(mu_);
  return engaged_;
}

std::array<double, 2> SmplIntentSource::LatestWristYaw() const
{
  std::lock_guard<std::mutex> lock(mu_);
  return wrist_yaw_;
}

bool SmplIntentSource::HasWristYaw() const
{
  std::lock_guard<std::mutex> lock(mu_);
  return has_wrist_yaw_;
}

std::array<double, 4> SmplIntentSource::LatestWristPr() const
{
  std::lock_guard<std::mutex> lock(mu_);
  return wrist_pr_;
}

bool SmplIntentSource::Window(double now_mono,
                              std::array<SmplFrame, SMPL_WINDOW>& out) const
{
  std::lock_guard<std::mutex> lock(mu_);
  if (ring_.size() < 4) return false;
  // Python: want = now - DELAY_S + k*SMPL_DT; idx = searchsorted(ts, want,
  // 'right') - 1, clipped -> the latest frame at or before each want time.
  for (std::size_t k = 0; k < SMPL_WINDOW; ++k) {
    const double want = now_mono - SMPL_DELAY_S + static_cast<double>(k) * SMPL_DT;
    std::size_t idx = 0;
    for (std::size_t i = 0; i < ring_.size(); ++i) {
      if (ring_[i].t_mono <= want) idx = i; else break;
    }
    out[k] = ring_[idx];
  }
  return true;
}

}  // namespace agi_x2
