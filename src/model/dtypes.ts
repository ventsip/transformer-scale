export const DTYPES = {
  fp32: { label: 'fp32', bytes: 4 },
  int64: { label: 'int64', bytes: 8 },
  int32: { label: 'int32', bytes: 4 },
  bf16: { label: 'bf16', bytes: 2 },
  fp8: { label: 'fp8', bytes: 1 },
  int4: { label: 'int4', bytes: 0.5 },
} as const;

export type DType = keyof typeof DTYPES;
