output "postgres_container_name" {
  description = "Name of the Postgres container"
  value       = docker_container.postgres.name
}

output "database_url" {
  description = "Local connection string (for debugging)"
  value       = "postgresql://${var.db_user}:${var.db_password}@localhost:5432/${var.db_name}"
  sensitive   = true
}
