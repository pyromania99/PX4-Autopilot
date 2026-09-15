/****************************************************************************
 *
 *   Copyright (c) 2026 PX4 Development Team. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 * 3. Neither the name PX4 nor the names of its contributors may be
 *    used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ****************************************************************************/

/**
 * @file SeatTest.cpp
 *
 * The seat is a rotation plus one scalar integral law, so these are properties rather
 * than recorded values. They mirror the level-1 checks in
 * examples/tools/evaluations/test_command_rotation.py (PegasusSimulator), because the
 * whole point of the ladder is that both rungs run the same law - a divergence between
 * the two files should surface here, not in a flight.
 *
 * What carries the weight:
 *  - the ROTATION, checked against R(theta) directly and against exact passthrough in
 *    the two cases that must cost nothing (mode 0, and theta == 0);
 *  - CONVERGENCE, asserted as its formula: a constant synthetic misalignment must decay
 *    exponentially and theta_s must reach the angle that produced it;
 *  - the SIGN, in both directions. Backwards it DOUBLES the misalignment, which crashed
 *    the vehicle at level 1, so the test asserts the error shrinks with the sign right
 *    AND grows with it flipped - a regression cannot pass by accident;
 *  - the TWO LAWS agreeing on where they converge, since that is what lets the cross law
 *    be the default without retuning the fixed point.
 */

#include <gtest/gtest.h>

#include "Seat.hpp"

#include <mathlib/mathlib.h>
#include <parameters/param.h>
#include <px4_platform_common/defines.h>

#include <cmath>

using namespace matrix;

namespace
{

/// updateParams() is protected; a test needs to re-cache after changing a parameter.
class TestSeat : public Seat
{
public:
	TestSeat() : Seat(nullptr) {}
	using Seat::updateParams;
};

void setF(const char *n, float v) { param_set_no_notification(param_find(n), &v); }
void setI(const char *n, int32_t v) { param_set_no_notification(param_find(n), &v); }

void resetParams()
{
	setI("MC_SEAT_MODE", 0);
	setI("MC_SEAT_LAW", 1);
	setI("MC_SEAT_KSIGN", 1);
	setF("MC_SEAT_K", 2.0f);
	setF("MC_SEAT_THMAX", 1.4f);
	setF("MC_SEAT_RMIN", 5.0f);
	setF("MC_SEAT_TAU_A", 0.002f);
	setF("MC_SEAT_T", 0.020f);
}

constexpr float kDt = 0.0025f;		///< 400 Hz, the rate_ctrl cadence
constexpr float kSpin = 30.f;		///< [rad/s] the SITL spin rate
constexpr float kMag = 3.f;		///< [rad/s^2] excitation magnitude

Vector2f rot(const Vector2f &v, float a)
{
	return Vector2f(cosf(a) * v(0) - sinf(a) * v(1), sinf(a) * v(0) + cosf(a) * v(1));
}

/**
 * Drive the seat against a synthetic plant under steady spin.
 *
 * THE EXCITATION is a roll/pitch command of constant magnitude rotating at the spin rate:
 * that is what an earth-frame tilt command looks like in a spinning body frame, and the
 * constant magnitude means the pair never starves, so the properties below are about the
 * law and not about the gate.
 *
 * THE PLANT is the claim under test and nothing else: whatever the seat emits is
 * delivered rotated by -psi. With the seat holding theta_s the measured response sits at
 * theta_s - psi from the command, so driving that to zero is driving theta_s to psi.
 *
 * The response is built from the angle the seat holds ENTERING the tick, leaving the
 * measurement one adaptation step behind - which is what the real loop does too.
 */
struct History {
	float theta_final{0.f};
	float alpha_first{NAN};
	float alpha_final{NAN};
	float theta_at(size_t i) const { return theta[i]; }
	float alpha_at(size_t i) const { return alpha[i]; }
	std::vector<float> theta;
	std::vector<float> alpha;
};

History runSpin(Seat &seat, float psi, int ticks, float r = kSpin, float dt = kDt)
{
	History h;

	for (int i = 0; i < ticks; i++) {
		const float t = i * dt;
		const Vector2f xd = rot(Vector2f(kMag, 0.f), r * t);
		const Vector2f xa = rot(xd, seat.theta() - psi);

		// Torque carries a distinctive yaw so a leak into it would show.
		seat.apply(Vector3f(xd(0), xd(1), 0.37f), xd, xa, r, dt);

		h.theta.push_back(seat.theta());
		h.alpha.push_back(seat.alpha());

		if (i == 0 || !PX4_ISFINITE(h.alpha_first)) {
			h.alpha_first = seat.alpha();
		}
	}

	h.theta_final = seat.theta();
	h.alpha_final = seat.alpha();
	return h;
}

class SeatTest : public ::testing::Test
{
public:
	void SetUp() override
	{
		// Without this the first param write blocks forever: autosave schedules onto
		// wq:lp_default, which the gtest harness does not start. Same trap as
		// TrajectoryStageTest.
		param_control_autosave(false);
		resetParams();
	}

	void TearDown() override { resetParams(); }
};

// --- rotation ----------------------------------------------------------------------

TEST_F(SeatTest, rotatesRollPitchByThetaAndNeverYaw)
{
	// Fixed mode makes theta_s a known function of the inputs: r*tau_a + atan(r*T).
	// With T at zero that is exactly r*tau_a, pinning theta_s at 0.5 rad without
	// needing any access to internals.
	setI("MC_SEAT_MODE", 1);
	setF("MC_SEAT_TAU_A", 0.02f);
	setF("MC_SEAT_T", 0.f);

	TestSeat seat;
	const float r = 25.f;

	// Four commands, so a transposed or sign-flipped matrix entry cannot hide.
	const Vector2f cmds[] = {{0.3f, 0.f}, {0.f, 0.3f}, {-0.2f, 0.4f}, {0.15f, -0.25f}};

	for (const Vector2f &tau : cmds) {
		const Vector3f out = seat.apply(Vector3f(tau(0), tau(1), 0.37f),
						Vector2f{}, Vector2f{}, r, kDt);
		const Vector2f want = rot(tau, 0.5f);

		EXPECT_NEAR(seat.theta(), 0.5f, 1e-5f);
		EXPECT_NEAR(out(0), want(0), 1e-6f);
		EXPECT_NEAR(out(1), want(1), 1e-6f);
		EXPECT_FLOAT_EQ(out(2), 0.37f) << "yaw torque must never be rotated";
	}
}

TEST_F(SeatTest, rotationPreservesMagnitude)
{
	// A rotation cannot change how much torque was asked for, only where it points.
	setI("MC_SEAT_MODE", 1);
	setF("MC_SEAT_TAU_A", 0.03f);
	setF("MC_SEAT_T", 0.f);

	TestSeat seat;
	const Vector2f tau(0.31f, -0.17f);

	for (float r = -30.f; r <= 30.f; r += 5.f) {
		const Vector3f out = seat.apply(Vector3f(tau(0), tau(1), 0.f),
						Vector2f{}, Vector2f{}, r, kDt);
		EXPECT_NEAR(Vector2f(out(0), out(1)).norm(), tau.norm(), 1e-6f);
	}
}

TEST_F(SeatTest, modeZeroIsExactPassthrough)
{
	// Not "close to" passthrough: bit-exact, because mode 0 returns before touching
	// anything. That is what makes the feature free when it is off.
	TestSeat seat;
	const Vector3f tau(0.11f, -0.42f, 0.37f);

	for (int i = 0; i < 100; i++) {
		const Vector3f out = seat.apply(tau, Vector2f(1.f, 2.f), Vector2f(2.f, -1.f),
						kSpin, kDt);
		EXPECT_FLOAT_EQ(out(0), tau(0));
		EXPECT_FLOAT_EQ(out(1), tau(1));
		EXPECT_FLOAT_EQ(out(2), tau(2));
	}

	EXPECT_FLOAT_EQ(seat.theta(), 0.f) << "and no state moved";
}

TEST_F(SeatTest, zeroAngleIsPassthrough)
{
	setI("MC_SEAT_MODE", 1);
	setF("MC_SEAT_TAU_A", 0.f);
	setF("MC_SEAT_T", 0.f);

	TestSeat seat;
	const Vector3f tau(0.11f, -0.42f, 0.37f);
	const Vector3f out = seat.apply(tau, Vector2f{}, Vector2f{}, kSpin, kDt);

	EXPECT_FLOAT_EQ(seat.theta(), 0.f);
	EXPECT_NEAR(out(0), tau(0), 1e-7f);
	EXPECT_NEAR(out(1), tau(1), 1e-7f);
}

// --- convergence --------------------------------------------------------------------

TEST_F(SeatTest, adaptiveConvergesToTheMisalignment)
{
	const float psi = 0.4f;
	setI("MC_SEAT_MODE", 2);
	setI("MC_SEAT_LAW", 0);		// angle law: k is dimensionless, decay is exp(-k t)
	setF("MC_SEAT_K", 2.0f);

	TestSeat seat;
	const History h = runSpin(seat, psi, 1600);	// 4 s

	// The angle finds a misalignment it was never told about.
	EXPECT_NEAR(seat.theta(), psi, 0.02f);
	EXPECT_NEAR(seat.alpha(), 0.f, 0.02f);
	EXPECT_NEAR(h.alpha_first, -psi, 0.05f) << "the initial misalignment IS the angle";

	/*
	 * alpha(t) = alpha(0) exp(-k t), asserted as the formula. The 15% band covers two
	 * known effects, both real in the vehicle too: the explicit Euler update
	 * approximates the exponential to O((k dt)^2) per step, and the measurement lags
	 * theta_s by one tick.
	 */
	for (const float t : {0.25f, 0.5f, 1.0f, 1.5f}) {
		const size_t i = (size_t)(t / kDt);
		const float predicted = h.alpha_first * expf(-2.0f * t);
		ASSERT_TRUE(PX4_ISFINITE(h.alpha_at(i)));
		EXPECT_NEAR(h.alpha_at(i), predicted, 0.15f * fabsf(h.alpha_first))
				<< "alpha(" << t << ")";
	}
}

TEST_F(SeatTest, bothLawsShareTheSameFixedPoint)
{
	// This is what lets CROSS be the default without retuning where the angle lands.
	// The gains differ because the cross drive carries |xd||xa| = kMag^2, so its k must
	// be scaled by 1/kMag^2 to match - which is exactly the dimensionality warning in
	// the MC_SEAT_LAW documentation, here made concrete.
	const float psi = 0.35f;

	setI("MC_SEAT_MODE", 2);
	setI("MC_SEAT_LAW", 0);
	setF("MC_SEAT_K", 2.0f);
	TestSeat angle;
	runSpin(angle, psi, 2400);

	setI("MC_SEAT_LAW", 1);
	setF("MC_SEAT_K", 2.0f / (kMag * kMag));
	TestSeat cross;
	cross.updateParams();
	runSpin(cross, psi, 2400);

	EXPECT_NEAR(angle.theta(), psi, 0.02f);
	EXPECT_NEAR(cross.theta(), psi, 0.02f);
	EXPECT_NEAR(angle.theta(), cross.theta(), 0.02f);
}

TEST_F(SeatTest, crossLawSuppressesItsOwnUpdateInAStarvedChannel)
{
	// The property the cross law exists for, and the reason it is the default: the step
	// scales with the signal magnitudes, so a quiet channel cannot drive theta_s far
	// however long it runs. The angle law normalises that away and keeps stepping.
	const float psi = 0.4f;
	const float tiny = 1e-3f;		// a nearly-silent channel

	setI("MC_SEAT_MODE", 2);
	setF("MC_SEAT_K", 2.0f);

	float moved[2];

	for (int law = 0; law < 2; law++) {
		setI("MC_SEAT_LAW", law);
		TestSeat seat;
		seat.updateParams();

		for (int i = 0; i < 4000; i++) {
			const float t = i * kDt;
			const Vector2f xd = rot(Vector2f(tiny, 0.f), kSpin * t);
			const Vector2f xa = rot(xd, seat.theta() - psi);
			seat.apply(Vector3f(xd(0), xd(1), 0.f), xd, xa, kSpin, kDt);
		}

		moved[law] = fabsf(seat.theta());
	}

	EXPECT_GT(moved[0], 10.f * moved[1])
			<< "angle law moved " << moved[0] << ", cross law " << moved[1];
	EXPECT_LT(moved[1], 0.05f) << "cross law should barely move on a silent channel";
}

// --- sign ---------------------------------------------------------------------------

TEST_F(SeatTest, signRightShrinksErrorSignFlippedGrowsIt)
{
	const float psi = 0.3f;
	setI("MC_SEAT_MODE", 2);
	setI("MC_SEAT_LAW", 0);
	setF("MC_SEAT_K", 2.0f);

	setI("MC_SEAT_KSIGN", 1);
	TestSeat right;
	const History rh = runSpin(right, psi, 1200);
	ASSERT_TRUE(PX4_ISFINITE(rh.alpha_first));
	EXPECT_LT(fabsf(rh.alpha_final), 0.1f * fabsf(rh.alpha_first));
	EXPECT_NEAR(right.theta(), psi, 0.03f);

	// Backwards, the angle runs AWAY and the error grows until the clamp catches it.
	// Asserting both directions is the point: a regression that dropped the sign
	// entirely would still pass the test above.
	setI("MC_SEAT_KSIGN", -1);
	TestSeat flipped;
	flipped.updateParams();
	const History fh = runSpin(flipped, psi, 1200);
	ASSERT_TRUE(PX4_ISFINITE(fh.alpha_first));
	EXPECT_GT(fabsf(fh.alpha_final), fabsf(fh.alpha_first));
	EXPECT_LT(flipped.theta(), 0.f) << "the wrong sign drives theta the wrong way";
	EXPECT_GT(fabsf(fh.alpha_final), psi) << "and leaves it worse than no seat at all";
}

// --- gates and limits ---------------------------------------------------------------

TEST_F(SeatTest, belowRminTheAngleIsHeld)
{
	const float psi = 0.4f;
	setI("MC_SEAT_MODE", 2);
	setI("MC_SEAT_LAW", 0);
	setF("MC_SEAT_RMIN", 5.f);

	TestSeat seat;
	runSpin(seat, psi, 1600);
	const float converged = seat.theta();
	ASSERT_NEAR(converged, psi, 0.03f);

	// Spin stops. A deliberately hostile measurement must not move the angle at all:
	// ungated, k=2 against a -1.0 rad error would reach the negative clamp well inside
	// this window.
	runSpin(seat, -1.f, 800, 1.f);
	EXPECT_FLOAT_EQ(seat.theta(), converged) << "held means held";
	EXPECT_FALSE(PX4_ISFINITE(seat.alpha())) << "gated means no measurement, not a stale one";
}

TEST_F(SeatTest, saturationFreezesTheAdaptation)
{
	const float psi = 0.4f;
	setI("MC_SEAT_MODE", 2);
	setI("MC_SEAT_LAW", 0);

	TestSeat seat;
	runSpin(seat, psi, 200);		// one 1/k, so it is moving but far from converged
	const float partial = seat.theta();
	ASSERT_GT(partial, 0.05f);
	ASSERT_LT(partial, psi - 0.05f);

	seat.setSaturated(true);
	runSpin(seat, psi, 1600);
	EXPECT_FLOAT_EQ(seat.theta(), partial) << "frozen means frozen, not slowed";

	seat.setSaturated(false);
	runSpin(seat, psi, 1600);
	EXPECT_NEAR(seat.theta(), psi, 0.03f) << "and it resumes where it left off";
}

TEST_F(SeatTest, clampedToThetaMaxAndBelowNinetyDegrees)
{
	setI("MC_SEAT_MODE", 2);
	setI("MC_SEAT_LAW", 0);
	setF("MC_SEAT_THMAX", 0.2f);

	TestSeat seat;
	runSpin(seat, 1.2f, 2000);
	EXPECT_NEAR(seat.theta(), 0.2f, 1e-4f);

	// Whatever is asked for, the angle cannot reach pi/2, where the authority a
	// rotation needs diverges as 1/cos(theta).
	setF("MC_SEAT_THMAX", 3.0f);
	TestSeat wide;
	wide.updateParams();
	runSpin(wide, 2.5f, 4000);
	EXPECT_LT(fabsf(wide.theta()), M_PI_F / 2.f);
}

TEST_F(SeatTest, shrinkingThetaMaxPullsTheLiveAngleIn)
{
	// A parameter change that tightens the limit takes effect immediately, or the angle
	// sits outside its own bound until the next adapting tick.
	setI("MC_SEAT_MODE", 2);
	setI("MC_SEAT_LAW", 0);
	setF("MC_SEAT_THMAX", 1.0f);

	TestSeat seat;
	runSpin(seat, 0.8f, 2400);
	ASSERT_GT(seat.theta(), 0.5f);

	setF("MC_SEAT_THMAX", 0.1f);
	seat.updateParams();
	EXPECT_NEAR(seat.theta(), 0.1f, 1e-5f);
}

TEST_F(SeatTest, degenerateInputsProduceNoNaN)
{
	setI("MC_SEAT_MODE", 2);
	TestSeat seat;

	for (int law = 0; law < 2; law++) {
		setI("MC_SEAT_LAW", law);
		TestSeat s;
		s.updateParams();

		// zero pair, NaN pair, zero dt - none may reach the mixer as non-finite
		for (int i = 0; i < 100; i++) {
			EXPECT_TRUE(s.apply(Vector3f(0.2f, 0.1f, 0.f), Vector2f{}, Vector2f{},
					    kSpin, kDt).isAllFinite());
			EXPECT_TRUE(s.apply(Vector3f(0.2f, 0.1f, 0.f), Vector2f(NAN, NAN),
					    Vector2f(NAN, NAN), kSpin, kDt).isAllFinite());
			EXPECT_TRUE(s.apply(Vector3f(0.2f, 0.1f, 0.f), Vector2f(1.f, 0.f),
					    Vector2f(0.f, 1.f), kSpin, 0.f).isAllFinite());
		}

		EXPECT_FLOAT_EQ(s.theta(), 0.f) << "law " << law << ": nothing observable, no adaptation";
	}
}

TEST_F(SeatTest, resetClearsTheAngle)
{
	// reset() runs on arm transitions, control-level changes and MC_CTRL_ALG changes. A
	// surviving angle is a step into the mixer at the worst possible moment.
	setI("MC_SEAT_MODE", 2);
	setI("MC_SEAT_LAW", 0);

	TestSeat seat;
	runSpin(seat, 0.4f, 1600);
	ASSERT_GT(fabsf(seat.theta()), 0.1f);

	seat.reset();
	EXPECT_FLOAT_EQ(seat.theta(), 0.f);
	EXPECT_FALSE(PX4_ISFINITE(seat.alpha()));
}

TEST_F(SeatTest, psiModelMatchesTheClosedForm)
{
	// Fixed mode's target, and the number a log should be read against.
	setI("MC_SEAT_MODE", 1);
	setF("MC_SEAT_TAU_A", 0.002f);
	setF("MC_SEAT_T", 0.020f);

	TestSeat seat;

	for (const float r : {13.f, 19.7f, 29.5f, -29.5f}) {
		EXPECT_NEAR(seat.psiModel(r), r * 0.002f + atanf(r * 0.020f), 1e-6f);
	}

	/*
	 * The level-1 reference point, and a real difference between the rungs: Pegasus
	 * models NO transport delay, only the first-order rotor lag. So level 1's angle at
	 * r = 29.5 is atan(29.5*0.020) = 0.533 rad, while PX4's default TAU_A = 2 ms puts
	 * the same airframe at 0.592. Comparing a PX4 seat_theta against a level-1 one
	 * without accounting for that is comparing two different models.
	 *
	 * Either way psiModel is a REFERENCE, not a target: the level-1 seat converged to
	 * 0.15 rad at this spin rate, a third of the model value, and flew well there.
	 */
	setF("MC_SEAT_TAU_A", 0.f);
	TestSeat lag_only;
	lag_only.updateParams();
	EXPECT_NEAR(lag_only.psiModel(29.5f), 0.5330f, 2e-3f) << "the level-1 (Pegasus) angle";
	EXPECT_NEAR(seat.psiModel(29.5f), 0.5920f, 2e-3f) << "PX4 default, with 2 ms transport";
}

} // namespace
