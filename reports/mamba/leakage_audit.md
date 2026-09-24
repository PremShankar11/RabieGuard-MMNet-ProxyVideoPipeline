# Stage 4 Data Leakage Audit Report

**Project:** Zero Rabies-MMNet  
**Stage:** Stage 4 — Temporal Mamba Behavior Model  
**Date:** 2026-09-23  
**Audit Status:** PASSED (Zero Leakage Detected)  

---

## 1. Audit Checkpoints Summary

| Checkpoint | Expected Condition | Audit Result | Status |
|---|---|---|---|
| **47 Official AK Train Clips Preserved** | Exactly 47 clips allocated | Exactly 47 clips (37 train / 10 val) | **PASSED** |
| **21 Official AK Test Clips Preserved** | Exactly 21 clips strictly held out | Exactly 21 clips | **PASSED** |
| **Validation Origin Constraint** | Validation clips drawn exclusively from the 47 train clips | 100% drawn from train pool | **PASSED** |
| **Train / Validation Overlap** | Zero intersection between Train and Val clips | Overlap = 0 clips | **PASSED** |
| **Train / Test Overlap** | Zero intersection between Train and Test clips | Overlap = 0 clips | **PASSED** |
| **Validation / Test Overlap** | Zero intersection between Val and Test clips | Overlap = 0 clips | **PASSED** |
| **Window Cross-Contamination** | No temporal window spans across multiple clips | Single-clip window slicing | **PASSED** |
| **Test Label Blindness** | No test labels used for class weights or training loss | Class weights fit strictly on train split | **PASSED** |
| **Test Normalization Isolation** | No test statistics used for feature normalization | Body-relative normalization only | **PASSED** |
| **Model Selection Blindness** | Test set evaluated strictly ONCE after freezing | Selected solely on validation F1 | **PASSED** |

---

## 2. Partition Clip Allocation
- **Training Clips (37):** `AFLPRFGA`, `BLUYUTEK`, `BUKSUFGA`, `CKFVZXGD`, `CUFGUGCS`, `EDHHMXGD`, `EHXIFXGD`, `EOFBMBME`, `EUKRNXGD`, `HPRSIXDO`, `ICCBOFGA`, `IXJLIXGD`, `JFPCUGCS`, `KAFGDXGD`, `KUZYPXGD`, `LDTFJXGD`, `MVDCSFGA`, `OOTCBXGD`, `ORBGIGCS`, `OTNOKXDO`, `PRXWIXGD`, `PWJTIFGA`, `QFMMRXGD`, `QOZTQGCS`, `RJALNXGD`, `SCAYVXGD`, `SFSRCXGD`, `TTOSQFGA`, `UXNFAXGD`, `VGALMFGA`, `WZUIAXGD`, `XAKUOXGD`, `YEYILXGD`, `YKJFXGCS`, `YLZBNFGA`, `YSWGBGCS`, `ZPJLBFGA`
- **Validation Clips (10):** `AWJEUGCS`, `DUZPVFGA`, `GWRPTXDO`, `JLQYOXGD`, `LKEERFGA`, `OQQRVXGD`, `RTCFXXDO`, `SMLRJXGD`, `VDDGCFGA`, `ZGNADXGD`
- **Held-Out Test Clips (21):** `AVFJUXDO`, `CSHALXDO`, `DCVGZXDO`, `DWOLCXDO`, `EBOJNFGA`, `EPQDGFGA`, `ETOKQXGD`, `EVQRFFGA`, `GICIEXDO`, `GNCBFXGD`, `IKAMHFGA`, `KURRZFGA`, `PEPZMXDO`, `PPOLHFGA`, `RAUURFGA`, `SKCIHXDO`, `SOXNEFGA`, `TNANEFGA`, `TVAUKXGD`, `USTTFFGA`, `ZPTIZXDO`

**Conclusion:** All partitions are completely disjoint at the fundamental video-clip level. Zero data leakage has occurred.
