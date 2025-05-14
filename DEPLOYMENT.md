# Deploying Face Liveness Detection to Cloud Server

This guide provides instructions for deploying the Face Liveness Detection application to your cloud server with IP address 185.32.84.81.

## Prerequisites

- Docker and Docker Compose installed on your server
- Git installed on your server
- Basic knowledge of Linux commands
- Access to your server via SSH

## Deployment Steps

### 1. Clone the Repository (Already Done)

You mentioned you've already cloned the repository to your cloud server.

### 2. Generate SSL Certificates (Optional but Recommended)

For camera access in modern browsers, HTTPS is required. Generate self-signed certificates:

```bash
cd /path/to/Silent-Face-Anti-Spoofing-Custom
./generate_ssl_certs.sh
```

This will create SSL certificates in the `ssl` directory.

### 3. Build and Start the Application

Run Docker Compose to build and start the application:

```bash
cd /path/to/Silent-Face-Anti-Spoofing-Custom
docker-compose up --build -d
```

The `-d` flag runs the containers in the background.

### 4. Access the Application

- HTTP: `http://185.32.84.81:9001`
- HTTPS (if configured): `https://185.32.84.81:9443`

**Note:** When using self-signed certificates, browsers will show a security warning. You'll need to accept the risk to proceed.

### 5. Troubleshooting

If you encounter the "Error accessing camera: navigator.mediaDevices is undefined" issue:

1. Make sure you're accessing the application via HTTPS
2. Check that your browser supports the MediaDevices API
3. Ensure you've accepted any permission prompts from the browser

### 6. Managing the Application

- Stop the application: `docker-compose down`
- View logs: `docker-compose logs -f`
- Restart the application: `docker-compose restart`

## Security Considerations

- For production use, consider obtaining a proper SSL certificate from Let's Encrypt or another certificate authority
- Review and configure firewall rules to only allow necessary ports (9001 for HTTP, 9443 for HTTPS)
- Consider setting up a reverse proxy like Nginx for additional security features

## Additional Configuration

If you need to modify the application settings:

1. Edit the `docker-compose.yml` file to change environment variables or port mappings
2. Rebuild and restart the application with `docker-compose up --build -d`
