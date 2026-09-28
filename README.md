# Image quantization codec

Codec prototype built for APP2 "Codage de l'information" (S7, GEI, Université
de Sherbrooke). It takes a raster image, normalizes it, then encodes and
decodes it with one of three quantization techniques, reporting the PSNR and
bit rate achieved.

## Running it

```bash
pip install numpy pillow matplotlib
python Solution.py
```

Every setting sits at the top of `Solution.py`:

| parameter | role |
|---|---|
| `Choix` | 1 = VQ, 2 = DPCM, 4 = DCT |
| `I_source` | image to process (`lenna.bmp`, `crest.bmp`, `cman.tif`, `irm.tif`, `mandrill.tif`) |
| `BLOC_L`, `BLOC_C`, `K_QV` | block shape and codebook size, for VQ |
| `NB_BITS`, `DENSITE` | bits per prediction error and assumed density, for DPCM |
| `TAILLE_BLOC`, `DEBIT_CIBLE` | block size and mean bits per coefficient, for DCT |

The script is split into two `# %%` cells, encoding and decoding, which can be
run separately in PyCharm or VS Code.

## Layout

| file | role |
|---|---|
| `Solution.py` | main script: full chain and result reporting |
| `convert.py` | 24-bit RGB to 8-bit greyscale |
| `reduce.py` | resize to 256 x 256 by bilinear interpolation |
| `QV_encode.py` / `QV_decode.py` | vector quantization (LBG algorithm) |
| `DPCM_encode.py` / `DPCM_decode.py` | differential quantization |
| `DCT_encode.py` / `DCT_decode.py` | discrete cosine transform quantization |
| `transmit.py` | physical layer simulation and bit budget |
| `computePSNR.py` | distortion measurement |

Encoders and decoders live in separate files, as the specification requires.
Each decoder imports from its encoder only what must be rigorously identical on
both sides: the predictor for DPCM, the block layout for VQ and DCT.

## Pipeline

```
I_source → convert → reduce → I_reduced → encoder → Data → decoder → I_decoded
                                                      ↑
                                               physical layer
```

`convert` and `reduce` normalize any input image, whatever its dimensions and
colour depth, into a 256 x 256 8-bit greyscale matrix. This already compresses:
a 512 x 512 RGB image drops from 786,432 to 65,536 samples before any coding
takes place.

## The simulated physical layer

`Data` maps a key `N` to a cell whose values will be coded on `N` bits, so they
must all fit in `[0, 2^N - 1]`. `transmit` checks that and returns the total
budget in bits, or a negative code naming the offending cell.

No data reaches the decoder except through `Data`. As a result, **anything the
decoder needs must be quantized and sent as metadata**, and its cost counts
towards the rate: the quantization step for DPCM, the codebook for VQ, the
allocation tables for DCT.

## The three techniques

### Vector quantization (`Choix = 1`)

The image is split into blocks, each treated as a vector, and the LBG algorithm
builds a codebook of `K` representative vectors by alternating nearest-neighbour
classification with centroid recomputation. The codebook is initialized by
splitting, which makes the result deterministic.

Only the indices are transmitted, plus the codebook as metadata. That is the
method's central trade-off: a richer codebook raises the PSNR but costs rate.

The decoder is a plain table lookup — the asymmetry typical of VQ, expensive to
encode and almost free to decode.

### Differential quantization (`Choix = 2`)

Each pixel is predicted from its already reconstructed neighbours, and only the
prediction error is quantized. The predictor is JPEG-LS's MED, which detects
edges instead of averaging across them.

The encoder holds a local decoder and predicts on the reconstruction, never on
the source: that is what keeps both loops in step. Any drift, however small,
would produce no visible error but would propagate through the whole image.

The encoder exploits that local decoder to pick its quantization step: it tries
several widenings of the step from table 9.3, measures the real distortion of
each and keeps the best. The chosen factor is never transmitted — it is absorbed
into the step value, already metadata — so this costs no bits.

### Discrete cosine transform (`Choix = 4`)

The image is split into 8 x 8 blocks, each transformed by a 2D DCT. The variance
of every coefficient position is computed across all blocks, then bits are
allocated proportionally to the logarithm of that variance. High frequencies,
near zero everywhere, receive no bits and are not transmitted at all.

Since coefficients do not share a common code width, they are spread across
several `Data` cells, one per allocation.

Two details drive the result:

- **Bits are redistributed by marginal gain.** After rounding and clamping, the
  allocation formula no longer matches the exact budget. Leftover bits must go
  where they buy the most, `var · 4⁻ᴮ`, rather than simply to the
  highest-variance coefficient: that criterion piles bits onto a few
  coefficients until they saturate and starves the rest.
- **Coefficients are assumed Laplacian, not uniform.** With the uniform row of
  table 9.3 the quantizer range stays at 3.46 σ whatever the bit count; the
  energy beyond is clipped and that error never shrinks, so the PSNR plateaus
  around 29 dB. The Laplacian row widens the range with the rate and is worth
  more than 12 dB here.

## Results

PSNR in dB, measured on the supplied images after normalization to 256 x 256
greyscale.

| image | VQ (4.63 bpp) | DPCM (4.00 bpp) | DCT (4.84 bpp) |
|---|---|---|---|
| `lenna.bmp` | **45.06** | 39.41 | 41.35 |
| `cman.tif` | **44.37** | 34.55 | 34.80 |
| `irm.tif` | **45.06** | 37.75 | 34.28 |
| `crest.bmp` | **42.70** | 36.29 | 37.07 |
| `mandrill.tif` | **44.83** | 40.92 | 43.18 |

Configurations: VQ on horizontal pairs with a 512-vector codebook, DPCM at 4
bits per prediction error, DCT on 8 x 8 blocks at 4.8 mean bits per coefficient.
All three stay under the 5 bits/pixel limit including metadata, and the full
chain runs in a few seconds.

Vector quantization wins on all five images: its codebook adapts to what each
image actually contains, where DPCM and DCT apply a fixed statistical model.

The other two fail on opposite images, which is telling about their mechanisms.
DCT is at its best on the mandrill, whose dense texture concentrates well in
frequency, and at its worst on the MRI. DPCM does the reverse: the MRI's large
uniform areas separated by sharp boundaries suit a causal predictor but compact
poorly in frequency.

## Origin

This is a Python port of the MATLAB skeleton supplied with the assignment. The
file hierarchy and the image naming (`I_source`, `I_reduced`, `I_encoded`,
`I_metadata`, `I_decoded`) are preserved. Two notable adaptations: MATLAB's
`Data{N}` cell array becomes a `Data[N]` dictionary, keys still being the bit
width; and functions return multiple outputs as a tuple.
