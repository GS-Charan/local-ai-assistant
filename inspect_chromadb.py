"""
ChromaDB Inspector - Check what's stored in your database
Simple script to view all collections and their contents
"""

import chromadb
from pathlib import Path
import config


def inspect_chromadb():
    """Inspect all ChromaDB collections and show contents"""
    
    print("=" * 60)
    print("🔍 ChromaDB Inspector")
    print("=" * 60)
    
    # Connect to ChromaDB
    db_path = config.CHROMA_DB_DIR
    print(f"\n📁 Database location: {db_path}")
    
    if not Path(db_path).exists():
        print("❌ ChromaDB directory doesn't exist yet!")
        print("   Run the app first to create it.")
        return
    
    client = chromadb.PersistentClient(path=str(db_path))
    
    # List all collections
    collections = client.list_collections()
    print(f"\n📚 Found {len(collections)} collection(s):")
    for col in collections:
        print(f"   - {col.name}")
    
    print("\n" + "=" * 60)
    
    # Inspect each collection
    for collection in collections:
        print(f"\n🗂️  Collection: {collection.name}")
        print("-" * 60)
        
        count = collection.count()
        print(f"   Total items: {count}")
        
        if count == 0:
            print("   (Empty)")
            continue
        
        # Get all items
        results = collection.get()
        
        print(f"\n   📝 Stored items:\n")
        
        for i, (doc_id, document, metadata) in enumerate(zip(
            results['ids'], 
            results['documents'], 
            results['metadatas']
        ), 1):
            print(f"   [{i}] ID: {doc_id}")
            print(f"       Content: {document}")
            print(f"       Metadata: {metadata}")
            print()
    
    print("=" * 60)
    print("✅ Inspection complete!")
    print("=" * 60)


def quick_search_test():
    """Quick test to search ChromaDB"""
    
    print("\n" + "=" * 60)
    print("🔍 Quick Search Test")
    print("=" * 60)
    
    client = chromadb.PersistentClient(path=str(config.CHROMA_DB_DIR))
    
    try:
        facts_collection = client.get_collection("user_facts")
        
        if facts_collection.count() == 0:
            print("\n❌ No facts stored yet!")
            return
        
        # Test search
        query = "programming"
        print(f"\n🔎 Searching for: '{query}'")
        
        results = facts_collection.query(
            query_texts=[query],
            n_results=3
        )
        
        if results['documents'] and results['documents'][0]:
            print(f"\n✅ Found {len(results['documents'][0])} result(s):\n")
            for i, doc in enumerate(results['documents'][0], 1):
                print(f"   {i}. {doc}")
        else:
            print("\n❌ No results found")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print("\n" + "=" * 60)


def clear_all_data():
    """⚠️ DANGER: Clear all ChromaDB data"""
    
    print("\n" + "!" * 60)
    print("⚠️  WARNING: This will delete ALL stored memories!")
    print("!" * 60)
    
    confirm = input("\nType 'DELETE' to confirm: ")
    
    if confirm != "DELETE":
        print("❌ Cancelled")
        return
    
    client = chromadb.PersistentClient(path=str(config.CHROMA_DB_DIR))
    
    collections = client.list_collections()
    
    for col in collections:
        print(f"🗑️  Deleting collection: {col.name}")
        client.delete_collection(col.name)
    
    print("\n✅ All data cleared!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ChromaDB Inspector Tool")
    print("=" * 60)
    print("\nOptions:")
    print("  1. Inspect database (view all data)")
    print("  2. Quick search test")
    print("  3. Clear all data (⚠️ DANGER)")
    print("  4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        inspect_chromadb()
    elif choice == "2":
        quick_search_test()
    elif choice == "3":
        clear_all_data()
    elif choice == "4":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")
        
    print()
