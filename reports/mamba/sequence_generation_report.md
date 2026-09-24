# Stage 4 Sequence Generation Report

**Project:** Zero Rabies-MMNet  
**Stage:** Stage 4 — Temporal Sequence & Window Construction  
**Date:** 2026-09-23  

---

## 1. Partition Breakdown (Clip & Window Level)

| Partition | Clips | Total Windows | CALM Windows | AGITATED Windows | Padded Windows | Total Valid Frames | Total Invalid Frames | Total Ambiguous Frames |
|---|---|---|---|---|---|---|---|---|
| **TRAIN** | 37 | 111 | 46 | 65 | 14 | 1,286 | 490 | 242 |
| **VAL** | 10 | 23 | 8 | 15 | 6 | 298 | 70 | 32 |
| **TEST** | 21 | 62 | 11 | 51 | 7 | 853 | 139 | 232 |
| **TOTAL** | 68 | 196 | 65 | 131 | 27 | 2,437 | 699 | 506 |

---

## 2. Sequence Construction Specifications
- **Sampling Rate:** 8.0 FPS (~125 ms temporal delta per step)
- **Sequence Length ($T$):** 16 observations (~2.0 seconds receptive field)
- **Frame Feature Vector:** 24 normalized keypoint coordinates $(x_{norm}, y_{norm}, conf_{kpt}) = 72$ dimensions
- **Window Stride:** 8 frames (50% temporal overlap for clips with $N \ge 16$ frames)
- **Short Clips Handling ($N < 16$ frames):** Edge-replication padding to $T=16$ with explicit boolean `padding_mask`
- **Reliability Weighting:** $\text{reliability}_t = \text{bbox\_conf}_t \times (\text{valid\_mask}_t \land \neg \text{padding\_mask}_t) \times (0.8 \text{ if ambiguous else } 1.0)$

---

## 3. Clip Partition Allocation

### Training Clips (37 clips):
`AFLPRFGA`, `BLUYUTEK`, `BUKSUFGA`, `CKFVZXGD`, `CUFGUGCS`, `EDHHMXGD`, `EHXIFXGD`, `EOFBMBME`, `EUKRNXGD`, `HPRSIXDO`, `ICCBOFGA`, `IXJLIXGD`, `JFPCUGCS`, `KAFGDXGD`, `KUZYPXGD`, `LDTFJXGD`, `MVDCSFGA`, `OOTCBXGD`, `ORBGIGCS`, `OTNOKXDO`, `PRXWIXGD`, `PWJTIFGA`, `QFMMRXGD`, `QOZTQGCS`, `RJALNXGD`, `SCAYVXGD`, `SFSRCXGD`, `TTOSQFGA`, `UXNFAXGD`, `VGALMFGA`, `WZUIAXGD`, `XAKUOXGD`, `YEYILXGD`, `YKJFXGCS`, `YLZBNFGA`, `YSWGBGCS`, `ZPJLBFGA`

### Validation Clips (10 clips):
`AWJEUGCS`, `DUZPVFGA`, `GWRPTXDO`, `JLQYOXGD`, `LKEERFGA`, `OQQRVXGD`, `RTCFXXDO`, `SMLRJXGD`, `VDDGCFGA`, `ZGNADXGD`

### Held-Out Test Clips (21 clips):
`AVFJUXDO`, `CSHALXDO`, `DCVGZXDO`, `DWOLCXDO`, `EBOJNFGA`, `EPQDGFGA`, `ETOKQXGD`, `EVQRFFGA`, `GICIEXDO`, `GNCBFXGD`, `IKAMHFGA`, `KURRZFGA`, `PEPZMXDO`, `PPOLHFGA`, `RAUURFGA`, `SKCIHXDO`, `SOXNEFGA`, `TNANEFGA`, `TVAUKXGD`, `USTTFFGA`, `ZPTIZXDO`

---

## 4. Integrity Assertion
- Zero clip overlap between Train, Validation, and Test partitions.
- No temporal window spans across multiple source video clips.
- Official Animal Kingdom 21 test clips strictly held out.
