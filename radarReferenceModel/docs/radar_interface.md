# Radar interface (radar node to the vision and fusion unit)

All radar messages are UTF-8 JSON objects, one per UDP datagram, sent to port 5005 on the
receiver (`--dest host:port` on `node.py`). Implemented in `radar_node/messages.py`.
Test receiver: `radar_node/listener.py`.

## Common fields

| Field | Type | Meaning |
|---|---|---|
| `type` | string | `radar_detection` or `radar_status` |
| `source` | string | Sender name, `radar-pynq` |
| `seq` | integer | Increases by one per message from this source. A gap means lost messages |
| `timestamp` | number | Unix time in seconds (0.1 ms): frame capture time for detections, send time for status |
| `payload` | object | Type-specific fields below |
| `checksum` | string | CRC-32 of the message without this field, as 8 lowercase hex digits |

The checksum is computed over the message serialized as compact JSON with sorted keys
(`separators=(",", ":")`). Missing values, such as NaN, are sent as `null`.

## `radar_detection` (one per frame, 20 per second)

| Field | Type | Meaning |
|---|---|---|
| `detected` | bool | A target was found by either detector |
| `range_m` | number or null | Range in metres |
| `speed_m_s` | number or null | Radial speed, approaching is negative. `null` for blade-channel-only detections |
| `snr_db` | number or null | Main detector: peak over median. Blade channel: blade-band excess in dB |
| `class` | string | `drone`, `person`, `unknown` or `none` |
| `confidence` | number | 0 to 1, fraction of recent frames supporting the class |
| `track_id` | integer | Main tracker ID (0 = no track), 1000 for blade-channel detections |
| `frame` | integer | Frame number since start |
| `latency_ms` | number | Capture to send time on the PYNQ |
| `detector` | string | `main`, `blade` or `main+blade` |
| `features` | object | Classifier and blade-channel features (below) |

`features`: `flash` (blade frame), `flash_hz` (estimated flash rate), `kurtosis`,
`doppler_spread_m_s`, `blade_state` (`warming`, `learning`, `running`), `blade_excess_db`,
`blade_excess_hi_db`, `blade_hits` (hits in the last 20 frames).

## `radar_status` (about once a second)

| Field | Meaning |
|---|---|
| `state` | `initializing`, `running`, `disconnected` (no frames for 2 s), `fault` or `stopped` |
| `fps`, `frames` | Frame rate and count since start |
| `sdk_errors` | Frames lost by the radar (FIFO overflow) |
| `skipped_frames` | Frames dropped because processing fell behind |
| `fft_backend` | `numpy` or `pl` |
| `proc_ms_mean`, `proc_ms_max` | Processing time over the last 40 frames |
| `detail` | Free text (source name or error message) |

## Receiver checks (FR12)

- **Malformed:** not JSON, missing a field, or a bad checksum. Discard.
- **Lost:** gaps in `seq` per source. Count, do not request again.
- **Stale:** `now - timestamp` above the fusion window (1 s in `listener.py`). Needs the clocks
  synchronized: run `radar_node/clock_sync.py` on the Mac after each PYNQ boot (or NTP later).

Measured Nov 25 after clock synchronization: 2,518 messages in 2 minutes, 0 lost, 0 malformed, 0 stale.
