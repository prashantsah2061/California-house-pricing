#!/usr/bin/env python3
"""
Script to generate a secure secret key for Flask application
"""
import secrets
import string

def generate_secret_key(length=32):
    """Generate a secure random secret key"""
    alphabet = string.ascii_letters + string.digits + string.punctuation
    return ''.join(secrets.choice(alphabet) for _ in range(length))

if __name__ == '__main__':
    secret_key = generate_secret_key()
    print("Generated Secret Key:")
    print(secret_key)
    print("\nTo use this key:")
    print("1. Set it as an environment variable:")
    print(f"   export SECRET_KEY='{secret_key}'")
    print("2. Or add it to your .env file:")
    print(f"   SECRET_KEY={secret_key}")
    print("\n⚠️  Keep this key secret and don't commit it to version control!") 