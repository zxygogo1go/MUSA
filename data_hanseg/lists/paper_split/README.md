# HaN-Seg Paper Split

Fixed inter-subject CT registration split for the 42-case HaN-Seg set.

- Test pairs: cases 01-10 as five adjacent moving/fixed pairs.
- Validation pairs: cases 11-18 and 20-21 as five adjacent moving/fixed pairs.
- Training IDs: all remaining cases, including case 19.

case_19 is kept out of validation/test because the public HaN-Seg files omit
`OAR_OpticChiasm`; keeping it in training avoids missing-label metrics while
preserving all 42 cases.
