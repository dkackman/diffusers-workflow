# MiniMax-H3 audio-VAE encode context: a hard-cut held slice vs a continuous encode (#1516, stage #1544)

lem, NVIDIA GeForce RTX 3090 24 GB, 2026-11-22. Commit `3152688` (the probe's first commit),
diffusers 0.44.0.dev0, torch 2.14.1+cu130. MiniMax-H3 snapshot `1cc7549f2cf0af8ff2f09b24f3c57303c8aa6317`
(cached; the probe runs `HF_HUB_OFFLINE=1`). The audio VAE loads bf16, the way dw's H3 templates load it.

## Question

dw holds a song slice cut hard at both ends (`DwH3HoldAudioStep`). Do the slice's head tokens (causal attention) and tail
tokens (convolution lookahead) come out wrong on dw's own encode path (`h3_hold_steps.encode_audio_span`) against a
continuous encode of the whole song? And does 1 s of pre-context (head), or 0.2 s of post-context (tail), bring them back?
The bar is the one #1516 plan v4 pins under "How the record reads the bar".

## Method

    python bench/h3_audio_vae_probe.py /home/don/diffusers-workspace/russian-anthem/assets/uploads/song_s23.mp3

The defaults are `--cuts 5 --slice-seconds 5`.

- **Song:** `russian-anthem/assets/uploads/song_s23.mp3` in Don's `russian-anthem` workspace, the master track of a held-song
  piece. It is 75.767 s long, 44.1 kHz stereo.
- **Normalizing:** the song is decoded at its own rate and normalized once, whole, with
  `MiniMaxH3Ref2VASetupStep._normalize_audio_condition`, onto 32 kHz stereo. That gives 3030 tokens, with a hop of
  800 samples (25 ms).
- **Reference:** one continuous `encode_audio_span` of the whole song.
- **Slices:** every slice and context window is cut from that same normalized waveform, on a hop boundary.
- **Cuts:**
  - one at 0 s;
  - three interior cuts, each the loudest slice head (RMS of its first 0.5 s) in its share of the song, with at least 2 s
    before it;
  - one whose slice ends at the song's last whole token.
- **Arms:** pre {0, 0.5, 1, 2} s × post {0, 0.2} s. The context's tokens are dropped after the encode.
- **Error:** per-token relative L2 against the same tokens of the continuous encode.
- **Floor:** the hard-cut arm's median error from 1 s into the slice to 1 s before its end.
- **Run columns:** the consecutive tokens strictly above 3× the floor, counted from the first token (head run) or the last
  (tail run).

## Result

Floors per cut:

| cut | start s | floor | 3× | 2× |
|---|---|---|---|---|
| 1 | 0.000 | 0.0003423 | 0.001027 | 0.0006846 |
| 2 | 2.000 | 0.003359 | 0.01008 | 0.006718 |
| 3 | 42.100 | 0.01208 | 0.03624 | 0.02416 |
| 4 | 50.550 | 0.00557 | 0.01671 | 0.01114 |
| 5 | 70.750 | 0.01406 | 0.04218 | 0.02812 |

| cut | start s | role | pre s | post s | tok0 rel | tok0 cos | head 0.25 | head 0.5 | last | tail 0.2 | head run s | tail run s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.000 | 0 s | 0 | 0 | 0.0003964 | 1 | 0.000513 | 0.0004315 | 0.6765 | 0.1613 | 0.000 | 0.225 |
| 1 | 0.000 | 0 s | 0 | 0.2 | 0.0003964 | 1 | 0.000513 | 0.0004313 | 0.005068 | 0.001054 | 0.000 | 0.050 |
| 1 | 0.000 | 0 s | 0.5 | 0 | n/a: no audio before |  |  |  |  |  |  |  |
| 1 | 0.000 | 0 s | 0.5 | 0.2 | n/a: no audio before |  |  |  |  |  |  |  |
| 1 | 0.000 | 0 s | 1 | 0 | n/a: no audio before |  |  |  |  |  |  |  |
| 1 | 0.000 | 0 s | 1 | 0.2 | n/a: no audio before |  |  |  |  |  |  |  |
| 1 | 0.000 | 0 s | 2 | 0 | n/a: no audio before |  |  |  |  |  |  |  |
| 1 | 0.000 | 0 s | 2 | 0.2 | n/a: no audio before |  |  |  |  |  |  |  |
| 2 | 2.000 | interior (loudest head) | 0 | 0 | 0.6346 | 0.7874 | 0.2873 | 0.176 | 0.5482 | 0.142 | 1.100 | 0.200 |
| 2 | 2.000 | interior (loudest head) | 0 | 0.2 | 0.6346 | 0.7874 | 0.2873 | 0.176 | 0.003486 | 0.001994 | 1.100 | 0.000 |
| 2 | 2.000 | interior (loudest head) | 0.5 | 0 | 0.02919 | 0.9996 | 0.0269 | 0.02201 | 0.5483 | 0.1422 | 0.825 | 0.200 |
| 2 | 2.000 | interior (loudest head) | 0.5 | 0.2 | 0.02919 | 0.9996 | 0.0269 | 0.02201 | 0.003709 | 0.001707 | 0.825 | 0.000 |
| 2 | 2.000 | interior (loudest head) | 1 | 0 | 0.00658 | 1 | 0.006919 | 0.006225 | 0.5485 | 0.1428 | 0.000 | 0.200 |
| 2 | 2.000 | interior (loudest head) | 1 | 0.2 | 0.00658 | 1 | 0.006919 | 0.006225 | 0.004422 | 0.001237 | 0.000 | 0.000 |
| 2 | 2.000 | interior (loudest head) | 2 | 0 | 0.0003119 | 1 | 0.0007769 | 0.0005891 | 0.5486 | 0.1431 | 0.000 | 0.200 |
| 2 | 2.000 | interior (loudest head) | 2 | 0.2 | 0.0003281 | 1 | 0.0006936 | 0.0005298 | 0.004604 | 0.0009952 | 0.000 | 0.000 |
| 3 | 42.100 | interior (loudest head) | 0 | 0 | 0.7117 | 0.782 | 0.2748 | 0.1578 | 0.6986 | 0.1851 | 0.375 | 0.150 |
| 3 | 42.100 | interior (loudest head) | 0 | 0.2 | 0.7117 | 0.782 | 0.2748 | 0.1578 | 0.01305 | 0.009502 | 0.375 | 0.000 |
| 3 | 42.100 | interior (loudest head) | 0.5 | 0 | 0.02988 | 0.9996 | 0.02119 | 0.01736 | 0.6983 | 0.1851 | 0.000 | 0.150 |
| 3 | 42.100 | interior (loudest head) | 0.5 | 0.2 | 0.02988 | 0.9996 | 0.02119 | 0.01736 | 0.01324 | 0.009814 | 0.000 | 0.000 |
| 3 | 42.100 | interior (loudest head) | 1 | 0 | 0.01312 | 0.9999 | 0.01149 | 0.01142 | 0.6983 | 0.1851 | 0.000 | 0.150 |
| 3 | 42.100 | interior (loudest head) | 1 | 0.2 | 0.01312 | 0.9999 | 0.01149 | 0.01142 | 0.01298 | 0.00965 | 0.000 | 0.000 |
| 3 | 42.100 | interior (loudest head) | 2 | 0 | 0.01141 | 0.9999 | 0.01183 | 0.01202 | 0.6982 | 0.1848 | 0.000 | 0.150 |
| 3 | 42.100 | interior (loudest head) | 2 | 0.2 | 0.01142 | 0.9999 | 0.01184 | 0.01203 | 0.01268 | 0.009021 | 0.000 | 0.000 |
| 4 | 50.550 | interior (loudest head) | 0 | 0 | 0.9797 | 0.6197 | 0.3087 | 0.1758 | 0.5841 | 0.1488 | 0.850 | 0.175 |
| 4 | 50.550 | interior (loudest head) | 0 | 0.2 | 0.9797 | 0.6197 | 0.3087 | 0.1758 | 0.007862 | 0.005869 | 0.850 | 0.000 |
| 4 | 50.550 | interior (loudest head) | 0.5 | 0 | 0.0284 | 0.9996 | 0.02226 | 0.01975 | 0.5841 | 0.149 | 0.250 | 0.175 |
| 4 | 50.550 | interior (loudest head) | 0.5 | 0.2 | 0.0284 | 0.9996 | 0.02226 | 0.01975 | 0.007893 | 0.006182 | 0.250 | 0.000 |
| 4 | 50.550 | interior (loudest head) | 1 | 0 | 0.01318 | 0.9999 | 0.01376 | 0.01374 | 0.584 | 0.1488 | 0.000 | 0.175 |
| 4 | 50.550 | interior (loudest head) | 1 | 0.2 | 0.01318 | 0.9999 | 0.01376 | 0.01374 | 0.00763 | 0.005959 | 0.000 | 0.000 |
| 4 | 50.550 | interior (loudest head) | 2 | 0 | 0.01507 | 0.9999 | 0.01457 | 0.01453 | 0.5843 | 0.1491 | 0.000 | 0.175 |
| 4 | 50.550 | interior (loudest head) | 2 | 0.2 | 0.01505 | 0.9999 | 0.01456 | 0.01452 | 0.007781 | 0.006248 | 0.000 | 0.000 |
| 5 | 70.750 | ends at song end | 0 | 0 | 0.7747 | 0.7373 | 0.351 | 0.2181 | 0.005541 | 0.008055 | 0.625 | 0.000 |
| 5 | 70.750 | ends at song end | 0 | 0.2 (silence past song end) | 0.7747 | 0.7373 | 0.351 | 0.2181 | 0.6971 | 0.2284 | 0.625 | 0.150 |
| 5 | 70.750 | ends at song end | 0.5 | 0 | 0.054 | 0.9986 | 0.04424 | 0.03817 | 0.005462 | 0.00777 | 0.150 | 0.000 |
| 5 | 70.750 | ends at song end | 0.5 | 0.2 (silence past song end) | 0.054 | 0.9986 | 0.04424 | 0.03817 | 0.697 | 0.2282 | 0.150 | 0.150 |
| 5 | 70.750 | ends at song end | 1 | 0 | 0.02755 | 0.9996 | 0.0245 | 0.02283 | 0.005289 | 0.00748 | 0.000 | 0.000 |
| 5 | 70.750 | ends at song end | 1 | 0.2 (silence past song end) | 0.02755 | 0.9996 | 0.0245 | 0.02283 | 0.6971 | 0.2281 | 0.000 | 0.150 |
| 5 | 70.750 | ends at song end | 2 | 0 | 0.01538 | 0.9999 | 0.01449 | 0.01414 | 0.005093 | 0.006786 | 0.000 | 0.000 |
| 5 | 70.750 | ends at song end | 2 | 0.2 (silence past song end) | 0.01543 | 0.9999 | 0.01449 | 0.01414 | 0.697 | 0.2277 | 0.000 | 0.150 |

**Determinism:** the encode is deterministic. Encoding cut 3's arm (pre 1 s, post 0) twice more gave a max |diff| of 0, and the
same head-0.25 (0.01149) and tail-0.2 (0.1851) errors both times.

**Verdict line** (the probe's own):

> cut 1 @ 0.000s: head a=no b=-, tail a=no b=-, does not meet; cut 2 @ 2.000s: head a=yes b=no, tail a=no b=-, does not meet;
> cut 3 @ 42.100s: head a=yes b=yes, tail a=no b=-, meets; cut 4 @ 50.550s: head a=yes b=no, tail a=no b=-, does not meet;
> cut 5 @ 70.750s: head a=yes b=yes, tail a=no b=-, meets; 2 of 5 cuts meet it: bar NOT MET

## Verdict

**The bar is not met:** only 2 of the 5 cuts meet it (cuts 3 and 5, both at the head), short of the 3 it needs.

## Reading (information beside the verdict, not part of it)

**The head error is real and large.**
- Every interior cut and the end cut show it. Token 0 has a cosine of 0.62–0.79 against the continuous encode, and the first
  0.25 s has a mean relative L2 of 0.27–0.35. The run above 3× the floor lasts 0.375–1.1 s.
- 1 s of pre-context removes most of it. It brings the first 0.25 s to 0.0069–0.025.
- It fails half (b) on cuts 2 and 4 by a small margin, against floors of 0.0034 and 0.0056: 0.006919 against a limit of
  0.006718, and 0.01376 against 0.01114. 2 s of pre-context gets cut 2 under its limit; cut 4's 2 s arm does not.

**The tail error is short.**
- On every interior cut the last token's error is 0.55–0.70, but the run above 3× the floor lasts only 0.15–0.225 s. That is
  under the 0.25 s half (a) needs, so no cut is judged at the tail.
- 0.2 s of post-context brings the last 0.2 s to about the floor on every interior cut (0.0020, 0.0095, 0.0059 against floors of 0.0034, 0.0121, 0.0056), well under 2× the floor.

**The end-of-song cut's post-silence arms err at the tail.** The continuous reference ends at the song's last token too, so
adding silence after it moves that cut's last tokens away from the reference. This doesn't affect the verdict, because the
hard-cut arm there has no tail run.

**Some floors are tiny for a reason.**
- Cut 1 starts where the continuous encode starts, so its head matches and its floor is tiny (0.00034).
- The floors of the interior cuts run from 0.0034 to 0.014. Neither arm's middle sees exactly the same context as the whole
  song, so the floor is not zero.

## What this record does not cover

- **One song, one run.** It is a 75.8 s anthem with a choral, dense mix. The "loudest head" picker is a loudness proxy for the
  plan's "inside vocal passages", not a vocal detector, and cut 2 landed on the earliest allowed start (2.0 s).
- **The tokens, not the result.** It measures the encode's tokens only. Whether the head error moves lip sync was not
  rendered, and was never in this stage's scope.
