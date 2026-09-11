terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0.0"
    }
  }
}

provider "docker" {}

resource "docker_image" "postgres" {
  name         = "postgres:${var.postgres_version}"
  keep_locally = true
}

resource "docker_volume" "postgres_data" {
  name = "predictive_maintenance_pgdata"
}

resource "docker_container" "postgres" {
  name  = "predictive-maintenance-postgres"
  image = docker_image.postgres.image_id

  env = [
    "POSTGRES_USER=${var.db_user}",
    "POSTGRES_PASSWORD=${var.db_password}",
    "POSTGRES_DB=${var.db_name}",
  ]

  ports {
    internal = 5432
    external = 5432
  }

  volumes {
    volume_name    = docker_volume.postgres_data.name
    container_path = "/var/lib/postgresql/data"
  }
}
