variable "postgres_version" {
  description = "Postgres image version tag"
  type        = string
  default     = "16-alpine"
}

variable "db_user" {
  description = "Database username"
  type        = string
  default     = "postgres"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
  default     = "postgres"
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "maintenance"
}
