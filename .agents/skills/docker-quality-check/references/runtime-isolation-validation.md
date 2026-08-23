# Runtime Isolation Validation

Use this procedure when a container contract claims a read-only filesystem,
restricted writable state, non-root execution, reduced privileges, or isolation from
the host.

## Define the boundary

Inventory every writable and authority-bearing surface before choosing flags:

- the image root filesystem, bind mounts, named volumes, tmpfs mounts, and device
  mounts;
- the effective UID/GID, supplementary groups, Linux capabilities, seccomp or other
  security profiles, and `no-new-privileges` state;
- network access and access to orchestration sockets or APIs;
- runtime-generated caches, compiled kernels, package state, logs, and outputs.

Prefer a read-only root filesystem and read-only inputs. Add only narrowly scoped,
bounded writable mounts needed by the documented workload. Use `noexec`, `nosuid`,
and `nodev` where the mount type and workload support them. Do not silently make a
cache executable to accommodate JIT compilation: review the compiler, generated-code
trust boundary, cache lifetime and size, and performance need, or select a non-JIT
path deliberately.

## Choose the runtime identity

Prefer a fixed unprivileged image identity when inputs can be provisioned once for
that identity. For direct host bind mounts, an explicit non-root host UID/GID can
preserve host ownership and read owner-only inputs, but it gives the workload that
host user's file authority. Record this weaker subject-separation tradeoff and never
assume UID/GID 1000.

Avoid recursive ownership changes on every startup. They require extra authority and
can be expensive for large or high-file-count trees. Prefer a dedicated volume,
one-time provisioning, or an explicit import/export handoff instead.

Treat access to a conventional rootful Docker daemon or its socket as
host-root-equivalent authority. A non-root container user, dropped capabilities, and
`no-new-privileges` constrain the workload but do not reduce the daemon's host
authority. Evaluate rootless Docker and user namespaces as separate host-boundary
designs rather than inferring them from the container UID.

## Prove the contract

Run the exact final image, entrypoint, mounts, identity, and security arguments used by
the documented workflow. Exercise a representative application operation, not only
`id`, a shell startup, or an import. Verify that:

- the process is non-root and has only the intended groups and capabilities;
- writes to the image root and read-only inputs fail;
- each intended writable path succeeds and remains within its size and execution
  policy;
- startup and the representative operation do not require an undeclared home,
  compiler cache, package cache, log path, or temporary directory;
- host-visible outputs have the intended ownership and mode;
- the container has no undeclared network, daemon socket, device, or secret access.

Record failures as contract defects. Do not weaken a boundary merely to make the smoke
test pass without updating the threat model and documented runtime contract.
