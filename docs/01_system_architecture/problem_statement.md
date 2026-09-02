# Problem Statement

- Initial Draft: 09/02/2026
- Latest Revision: 09/02/2026
- Authors: AJ Donald
- Status: Draft (pending team + supervisor review)

---

## 1. Problem Statement

Vehicle thefts, break-ins and component theft remain an ever growing and persistent issue all throughout Ontario.
Whether it be for homeowners with garages, carports, covered parking or even for those who park in common spaces, 
almost everyone knows someone who has had their car stolen. Some common preventative measures include but are not limited to, 
PIR-based driveway motion lights, camera based smart security systems (Ring/Nest) and reactive vehicle alarms. Each of these technologies
have specific, well-known blind spots or pitfalls. For instance, PIR sensors give only coarse, binary motion detection with no directionality or velocity information. 
As a result, these sensors can't reliably tell a person apart from wind-blown debris, the passing of an animal or even simple temperature drafts apart from real threats. 
These faults resulting in a considerable amount of false alarms which desensitize home-owners. Next, camera-based motion detection degrades significantly in accurancy
under low-light and dark conditions - exactly when most break-ins occur. Moreover, camera-based systems are equally prone to false triggers from headlight glare and moving shawdows / foliage.
Importantly, vehicle alarms and immobilizers offer users with some peace of mind, however these mitigation strategies are purely reactive - as they only trigger after physical contact 
or tampering. Thus, they fail to offer any early warning that someone is attempting to steal the vehicle, while also failing to provide any evidence or visual record of the perpetrator. 
In the market, no common consumer solution combines radar-based motion & velocity sensing, which is largely lighting-independent, with camera-based visual confirmation, despite this combination 
being well established in higher-end commerical and automative contexts, such as factories and dealerships. 

## 2. Existing Solutions and Their Limitations

- **PIR motion sensors / driveway lights** — inexpensive but only detect presence, with no velocity
  or directional data, easily triggered by non-human motion.
- **Camera-only smart security systems** (Ring/Nest/Arlo-style) — provides video and motion
  alerts, but rely on frame-differencing or on-device vision models that perform poorly at night and
  generate false positives from lighting artifacts.
- **Vehicle alarms / immobilizers** — detect only physical tampering after the fact, relies on human intervention and provides no
  perimeter awareness or early warning.
- None of these solutions combine an independent, lighting-robust motion/velocity sensor with a camera to
  cross-validate detections before alerting.

## 3. Proposed Solution

Consumers ultimately are looking for a system which provides them with confidence in the protection of their home and vehicle. 
A radar-assisted vision tracking system, mounted to monitor the space around a parked vehicle, whether it be in a garage, carport or covered area will provide consumers such peace of mind. 
An FPGA performes real-time DSP & hardware acceleration on the raw analog IF output of a Doppler radar module, which includes (DC removal, FIR filtering, windowing, FFT, peak detection and velocity estimation)
to detect the presence, motion and velocity of a human sized target, approaching or moving near the vehicle. Importantly, a radar detection gates a camera-based tracking pipeline, which visually 
locates and tracks the target within frame and records a timestamped video. Next, a fusion stage combines the two into a single tracked-object output with a confidence estimate, which only raises alerts 
when both sensors agree on a person-scale object is present and moving near the vehicle, which will reject false triggers that plague either sensor individually.


## 4. Why This Is a Good Solution

- **Lighting independence** — radar is mostly unaffected by darkness, which directly addresses the
  camera's weakest point at the time break-ins are most likely.
- **False-alarm reduction** — requiring agreement/handshake between an independent radar detection and a visual
  track significantly reduces the false-alarm rate that undermines PIR- and camera-only systems today.
- **Earlier warning** — radar can flag an approaching target before being well captured by the camera or
  before any physical contact occurs, giving an earlier warning window than contact-based alarms.
- **Engineering depth** — doing the Doppler signal processing at the raw analog-IF level in FPGA
  logic, rather than relying on a radar module that outputs pre-processed detections, is real
  hardware/DSP engineering work (fixed-point pipeline design, streaming architecture, verification),
  which is what makes this a legitimate FPGA capstone rather than an off-the-shelf smart-camera
  product.
- **Feasibility** — reuses work already in progress: the radar reference model and DSP algorithms
  (including multi-target detection) are already partially validated, the component list (HB100, ADC,
  Zybo Z20, Pcam 5C) fits the <$500 budget, and the subsystem breakdown (radar/FPGA, vision, fusion,
  integration) still maps cleanly onto the existing team roles.

---

## Notes / Open Items (for later refinement)

- Derive concrete technical requirements from this statement (detection range, false-alarm rate
  target, response latency, FOV, etc.) — needed before subsystem interfaces can be finalized.
- Reconcile with `docs/00_project_management/team_role.md` and `roadmap.md` per Rose's feedback
  (workload rebalancing, rotating PM responsibilities).
- Reply to Rose summarizing the updated direction.
