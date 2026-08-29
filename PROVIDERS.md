# Optional providers

The provider modules are retained for future enrichment but are intentionally separate from the default Axiom loop.

- **Helius**: forward on-chain/token observations.
- **Shyft**: cheap short-history parsed-transaction prescreen for unknown wallets.
- **Birdeye**: selective historical trade bootstrap with local usage circuit breakers.

Birdeye local defaults retained from the previous design:

```text
monthly maximum  30,000 CU
daily soft          900 CU
daily hard        1,000 CU
```

Provider usage is explicit. `run_axiom_*.bat` never calls these modules.
