terraform {
  # 1.10+ is required for the S3 backend's use_lockfile (see backend.tf.example).
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0, < 7.0"
    }
  }
}
