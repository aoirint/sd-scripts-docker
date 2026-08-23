---
name: docker-quality-check
description: >-
  Quality-check Dockerfiles, Compose configurations, container startup scripts,
  image CI, and container documentation. Use when creating, editing, or reviewing
  container build, runtime, orchestration, publication, or explicitly authorized
  registry image deletion changes.
---

# Docker Quality Check

## When to Use

- Use for changes to Dockerfiles, Compose files, container entrypoints, image build
  automation, registry publication or deletion, or container-facing documentation.
- Use before committing or publishing container changes.

## Goals

- Keep container builds reproducible, minimal, and suitable for the intended runtime.
- Validate both Dockerfile syntax and the changed build or Compose behavior.
- Make third-party images, downloaded tools, and CI actions traceable and reviewable.
- Keep notices for software distributed in the image visible and distinct from
  development-only dependency notices.

## Workflow

1. Read the changed container files and repository guidance. Identify the build targets,
   runtime user, exposed services, mounted paths, environment variables, and expected
   startup behavior.
2. Run the repository's documented container checks. When no project-specific command
   exists, use the Dockerfile frontend's official build checks (`docker buildx build
   --check`) for each changed Dockerfile and `docker compose config` for each changed
   Compose configuration. Pass the same context, file, target, platform, and build
   arguments as the affected build contract. Treat hadolint as an optional additive
   check only when the repository intentionally relies on its ShellCheck or package
   policy rules and the selected hadolint parser supports the Dockerfile syntax.
3. Build the affected image or target with `docker build` or `docker compose build`.
   Pass only the build arguments and secrets required by the documented build contract;
   never place credentials in image layers, build logs, or committed configuration.
4. When startup, routing, health checks, or service wiring changes, start the smallest
   affected service set and exercise its documented health endpoint or smoke check.
   Stop the test services after verification.
5. When the image adds or changes a GPU runtime, framework wheel, inference provider,
   or compiled accelerator extension, follow
   [accelerator-runtime-validation.md](references/accelerator-runtime-validation.md).
   Test every affected consumer with an actual accelerator operation in the final image,
   inspect dynamic-library resolution, reject unintended provider fallback, and exercise
   the documented application path. Do not infer compatibility from a build, import,
   version string, or device-visibility check.
6. Inspect image and runtime safety: use a non-root user where feasible, keep the final
   image free of build-only tooling and secrets, define a clear entrypoint, and avoid
   mutable base-image tags when an immutable digest is practical. When filesystem,
   identity, capability, or daemon isolation is part of the runtime contract, follow
   [runtime-isolation-validation.md](references/runtime-isolation-validation.md) and
   smoke-test the exact documented container arguments.
7. Inventory third-party software copied, installed, linked, or otherwise distributed in
   the final image. Put notices for the primary bundled application and other shipped
   runtime content at the top of `THIRD_PARTY_NOTICES.md`, before build tools, CI Actions,
   Agent Skills, or other development-only dependencies. For each primary bundled
   application, record its source, bundled location, version source, and license. Verify
   that required upstream license and notice files remain in the final image. Add a
   README disclosure that names the application, version source, and license and links to
   both `THIRD_PARTY_NOTICES.md` and upstream license information. Mark unavailable
   version, license, or final-image evidence as unverified rather than inferring a pass.
8. For newly introduced or updated external images, downloaded executables, or GitHub
   Actions, use `security-check` to assess provenance, version or digest pinning,
   release age, checksums, permissions, and runtime behavior. Pin GitHub Actions to
   full commit SHAs with accurate version comments. Use
   `github-actions-quality-check` for workflow structure, permissions, runners,
   validation, and publication gates. For an APM-managed repository, apply
   `apm-workflow` and keep `apm audit --ci` in the outer source-check action.
   Keep Markdown source validation in that same action so container-only
   changes cannot bypass the repository documentation gate.
9. Summarize commands run, build and smoke-test results, and every skipped check with a
   concrete reason.
10. For an explicitly authorized registry image deletion, follow
   [registry-image-deletion.md](references/registry-image-deletion.md). Treat registry
   state as the source of truth: resolve exact targets before mutation, account for
   shared manifests and registry-specific deletion units, and verify retained and
   removed references independently afterward. This procedure does not define a
   retention policy or recommend when deletion should occur.

## Dockerfile Check Selection

Prefer BuildKit build checks because they use the Dockerfile frontend selected by the
repository and validate build options as well as the file. In GitHub Actions, use
`docker/build-push-action` with `call: check`, or the equivalent
`docker buildx build --check` command after setting up Buildx. A check invocation does
not execute the image build, so keep a real build as a separate integrated-source gate.

Do not silently substitute hadolint when BuildKit checks are available. Hadolint parses
the Dockerfile independently and can reject supported frontend syntax before its rules
run. When a repository adds hadolint for complementary rules, document that purpose,
verify syntax compatibility, and pin the version and exact asset SHA-256 under the
repository's adoption policy.

## Default Checks

```shell
docker buildx build --check .
docker build -t local-validation .
docker compose config
```

Replace these examples with the repository's documented file paths, build targets, tags,
and Compose files. Do not treat a successful syntax check as evidence that the image
builds or starts correctly.

## CI Templates

Read [ci-template-contract.md](references/ci-template-contract.md) before creating or
repairing Docker CI. The bundled files under `assets/github/` keep pull-request checks
limited to source and BuildKit validation and reserve image builds for the exact integrated
main-branch commit.
Apply `github-actions-quality-check` for shared event, permission, runner, pinning, and
workflow behavior. Apply `github-workflow` for repository settings and enforcement.

## Resources

- [accelerator-runtime-validation.md](references/accelerator-runtime-validation.md): final-image
  runtime, linkage, provider, and application-path gates for GPU dependency changes.
- [runtime-isolation-validation.md](references/runtime-isolation-validation.md): read-only
  filesystems, writable surfaces, runtime identity, and daemon-boundary checks.
- [registry-image-deletion.md](references/registry-image-deletion.md): authorized registry
  deletion planning and verification.
