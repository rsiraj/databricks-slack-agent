# Pinned by digest so rebuilds are reproducible; the tag is kept for readability.
# Digest is for the linux/amd64 image (Lambda functions use architectures = ["x86_64"]).
# The Python minor is pinned deliberately (pyproject.toml requires-python and uv.lock
# are resolved for 3.12); .github/dependabot.yml only lets digest updates through.
FROM public.ecr.aws/lambda/python:3.12@sha256:1eb923b5c41f678871365b8f1bd3a469a4af1608e5d38f778cdd3b264051f096

# Pin pip itself so the resolver/hash-checking behaviour does not drift between builds.
RUN pip install --no-cache-dir "pip==26.2.1"

# Dependencies are resolved once, in uv.lock (from pyproject.toml). At build time uv
# exports the runtime (non-dev) subset of that lock as a hashed requirements file and
# pip installs it with --require-hashes, so every wheel is verified against the lock
# and there is no companion export file to keep in sync when uv.lock changes.
# `uv` is only needed for the export and is removed again in the same layer.
# Keep UV_VERSION in step with the workflows in .github/workflows/.
ARG UV_VERSION=0.10.6
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir "uv==${UV_VERSION}" \
 && uv export --frozen --no-dev --no-emit-project --format requirements.txt \
      -o /tmp/requirements.lock \
 && pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock --target ./ \
 && pip uninstall -y uv \
 && rm -f /tmp/requirements.lock

COPY receiver.py worker.py ./
COPY tools/ ./tools/
COPY utils/ ./utils/
COPY prompts/ ./prompts/
COPY config/ ./config/

# Default command; each Lambda overrides via ImageConfig.Command (managed by Terraform)
CMD ["worker.lambda_handler"]
