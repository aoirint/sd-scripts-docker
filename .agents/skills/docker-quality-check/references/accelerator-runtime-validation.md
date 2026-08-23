# Accelerator Runtime Validation

Use this procedure when a container adds or changes CUDA, ROCm, a GPU framework wheel,
an inference provider, a compiled accelerator extension, or supported GPU architecture.

## Establish the compatibility model

Inventory each independently loaded or executed consumer and its intended fallback policy.
Distinguish these layers:

- The host driver and device interface exposed to the container.
- User-space runtime libraries supplied by the base image or installed system packages.
- Libraries bundled by framework wheels or other language packages.
- Native extensions and provider libraries loaded at application runtime.

Version labels across these layers do not need to match. For example, a framework wheel
may bundle one CUDA generation while another provider dynamically requires a different
SONAME from the runtime image. Accept a mixed stack only when the final image resolves the
intended libraries and every affected consumer executes successfully.

## Validate the final image

1. Test the exact final image, preferably by immutable digest, on a representative GPU and
   supported driver. Record the image identity, GPU model, driver, platform, runtime user,
   package versions, and production-relevant container arguments.
2. Inspect every affected native provider or extension in the final image with suitable
   loader tooling such as `readelf`, `ldd`, or platform equivalents. Fail on unresolved
   dependencies or unintended library shadowing. Use loader diagnostics when ordinary
   linkage output does not establish which library is selected at runtime.
3. Execute an actual accelerator operation for every independently updated consumer.
   Synchronize asynchronous work and assert device placement, output shape or state change,
   and finite or otherwise meaningful results. A successful operation in one framework does
   not validate another framework, provider, or native extension.
4. For provider-based inference, request the intended accelerator provider explicitly and
   run a representative supported graph. Treat provider initialization errors, warnings,
   silent CPU selection, or execution solely through an unintended fallback as failure.
   Capture profiling or verbose placement evidence when a provider list alone cannot prove
   where representative nodes executed.
5. Run the repository's documented entrypoint or application workflow with a representative
   model, checkpoint, and input when that path is part of the support contract. Assert a
   meaningful output and inspect the child-process logs for loader or fallback warnings.

Imports, builds, version output, driver utilities, and device-visibility checks remain useful
diagnostics, but establish neither kernel execution nor cross-library ABI compatibility.
A driver utility missing from `PATH` proves only that command resolution failed; inspect the
platform's device and integration paths before declaring the accelerator unavailable, then retain
the actual-operation requirement above.

## Handle unavailable evidence

Keep passed, failed, and unavailable evidence separate. Name the missing GPU, driver,
provider instrumentation, model, checkpoint, input, or application command rather than
inferring a pass. When the release claims compatibility with the untested path, treat the
missing execution evidence as a release blocker. Otherwise disclose the limitation and
obtain explicit maintainer acceptance before publishing a narrower support claim.
