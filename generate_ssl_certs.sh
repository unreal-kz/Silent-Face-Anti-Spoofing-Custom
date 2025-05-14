#!/bin/bash
# Script to generate self-signed SSL certificates for development/testing

# Create SSL directory if it doesn't exist
mkdir -p ssl

# Generate a self-signed certificate valid for 365 days
openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes -subj "/CN=localhost" -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:185.32.84.81"

# Set permissions
chmod 600 ssl/key.pem
chmod 644 ssl/cert.pem

echo "Self-signed SSL certificates generated in the ssl directory."
echo "Note: These are self-signed certificates and will show security warnings in browsers."
echo "For production, consider using Let's Encrypt or another certificate authority."
