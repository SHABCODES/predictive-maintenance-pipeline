# Local Postgres Provisioning

This Terraform module uses the `kreuzwerker/docker` provider to provision a local PostgreSQL database container for the Predictive Maintenance API.

## Usage

```bash
terraform init
terraform plan
terraform apply
```

*Note: In a real production environment, Terraform state would be stored remotely (e.g., in an S3 bucket or Terraform Cloud), and this module would provision a managed database like AWS RDS or GCP Cloud SQL. This local Docker implementation demonstrates the IaC workflow without requiring cloud credentials or incurring costs.*
