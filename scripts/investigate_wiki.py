from huggingface_hub import HfApi

def inspect_wiki():
    api = HfApi()
    repo_id = "wikimedia/wikipedia"
    print(f"🔍 Inspecting {repo_id}...")
    
    try:
        # List files in the repo
        files = api.list_repo_files(repo_id, repo_type="dataset")
        
        # Filter for Vietnamese parquet files
        vi_files = [f for f in files if ".vi" in f and f.endswith(".parquet")]
        
        print(f"Found {len(vi_files)} Vietnamese Parquet files:")
        for f in vi_files[:10]:
            print(f" - {f}")
            
        if not vi_files:
            print("❌ No '.vi' parquet files found directly. Maybe hidden in subdirs?")
            # Print top 10 files
            for f in files[:10]:
                print(f" - {f}")
                
    except Exception as e:
        print(f"❌ API Error: {e}")

if __name__ == "__main__":
    inspect_wiki()
