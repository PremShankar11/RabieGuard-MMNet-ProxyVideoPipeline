# Video V2 Data Leakage Audit Report

**Project:** Zero Rabies-MMNet  
**Stage:** Video V2 Behavioral Model Development  
**Date:** 2026-09-23  
**Audit Status:** PASSED (Zero Data Leakage Detected)  

---

## 1. Audit Checkpoints

| Checkpoint | Expected Condition | Audit Result | Status |
|---|---|---|---|
| **47 Dev Clips Preserved** | Exactly 47 clips used in CV and V2 development | Exactly 47 clips | **PASSED** |
| **21 Test Clips Locked** | 21 test clips 100% excluded from V2 CV & selection | Zero test clips in CV | **PASSED** |
| **Fold Isolation** | Zero clip overlap across any of the 5 CV folds | Inter-fold overlap = 0 clips | **PASSED** |
| **Window Splitting** | No temporal window spans multiple source clips | Single-clip windowing | **PASSED** |
| **Normalization Blindness** | No test or validation statistics used in normalization | Body-relative normalization only | **PASSED** |
| **Class-Weight Blindness** | Class weights computed strictly from training partition | Fold-specific pos_weights | **PASSED** |
| **Selection Blindness** | V2 model selected strictly on development CV metrics | Test set untouched | **PASSED** |

---

## 2. Partition Clip Allocation
- **Development Set (47 Clips):** `AFLPRFGA`, `AWJEUGCS`, `BLUYUTEK`, `BUKSUFGA`, `CKFVZXGD`, `CUFGUGCS`, `DUZPVFGA`, `EDHHMXGD`, `EHXIFXGD`, `EOFBMBME`, `EUKRNXGD`, `GWRPTXDO`, `HPRSIXDO`, `ICCBOFGA`, `IXJLIXGD`, `JFPCUGCS`, `JLQYOXGD`, `KAFGDXGD`, `KUZYPXGD`, `LDTFJXGD`, `LKEERFGA`, `MVDCSFGA`, `OOTCBXGD`, `OQQRVXGD`, `ORBGIGCS`, `OTNOKXDO`, `PRXWIXGD`, `PWJTIFGA`, `QFMMRXGD`, `QOZTQGCS`, `RJALNXGD`, `RTCFXXDO`, `SCAYVXGD`, `SFSRCXGD`, `SMLRJXGD`, `TTOSQFGA`, `UXNFAXGD`, `VDDGCFGA`, `VGALMFGA`, `WZUIAXGD`, `XAKUOXGD`, `YEYILXGD`, `YKJFXGCS`, `YLZBNFGA`, `YSWGBGCS`, `ZGNADXGD`, `ZPJLBFGA`
- **Locked Test Set (21 Clips):** `AVFJUXDO`, `CSHALXDO`, `DCVGZXDO`, `DWOLCXDO`, `EBOJNFGA`, `EPQDGFGA`, `ETOKQXGD`, `EVQRFFGA`, `GICIEXDO`, `GNCBFXGD`, `IKAMHFGA`, `KURRZFGA`, `PEPZMXDO`, `PPOLHFGA`, `RAUURFGA`, `SKCIHXDO`, `SOXNEFGA`, `TNANEFGA`, `TVAUKXGD`, `USTTFFGA`, `ZPTIZXDO`

**Conclusion:** Full partition isolation is rigorously verified. V2 model selection was completed with zero test set leakage.
