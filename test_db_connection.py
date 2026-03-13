import psycopg2

# Try to connect
try:
    # On Mac (no password usually)
    import os
    pwd = os.environ.get("PGPASSWORD", "")
    conn = psycopg2.connect(
        dbname="pharma_supply_chain",
        user="postgres",  # or your username
        password=pwd,
        host="localhost",
        port="5433"
    )

    print("✅ Successfully connected to PostgreSQL!")

    # Test pgvector
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
    result = cursor.fetchone()

    if result:
        print("✅ pgvector extension is installed!")
    else:
        print("⚠️  pgvector extension NOT found")
        print("   Run: psql pharma_supply_chain -c 'CREATE EXTENSION vector;'")

    cursor.close()
    conn.close()

except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\nTroubleshooting:")
    print("1. Is PostgreSQL running? (brew services list)")
    print("2. Does database exist? (psql -l)")
    print("3. Check your username/password")
