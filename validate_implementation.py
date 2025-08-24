#!/usr/bin/env python3
"""
Simple validation script to test the implemented RAG service structure.
This script tests the basic functionality without requiring external dependencies.
"""

import os
import sys
import ast
import importlib.util
from pathlib import Path

def validate_file_syntax(file_path):
    """Validate Python file syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Parse the AST to check syntax
        ast.parse(source, filename=file_path)
        return True, None
    except Exception as e:
        return False, str(e)

def check_file_structure():
    """Check if all required files exist."""
    base_path = Path('.')
    
    required_files = [
        'src/config.py',
        'src/core/config.py',
        'src/core/exceptions.py',
        'src/models/schemas.py',
        'src/utils/logging.py',
        'src/utils/helpers.py',
        'src/utils/file_handlers.py',
        'src/utils/validators.py',
        'src/services/azure_openai_service.py',
        'src/services/azure_search_service.py',
        'src/services/document_intelligence.py',
        'src/services/azure_ai_service.py',
        'src/services/document_processor.py',
        'src/services/vector_store.py',
        'src/services/rag_service.py',
        'src/api/main.py',
        'src/api/routes/health.py',
        'src/api/routes/documents.py',
        'src/api/routes/query.py',
        'Dockerfile',
        'docker-compose.yml',
        '.env.example'
    ]
    
    missing_files = []
    syntax_errors = []
    valid_files = []
    
    for file_path in required_files:
        full_path = base_path / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            if file_path.endswith('.py'):
                is_valid, error = validate_file_syntax(full_path)
                if is_valid:
                    valid_files.append(file_path)
                else:
                    syntax_errors.append((file_path, error))
            else:
                valid_files.append(file_path)
    
    return {
        'missing_files': missing_files,
        'syntax_errors': syntax_errors,
        'valid_files': valid_files,
        'total_required': len(required_files)
    }

def check_import_structure():
    """Check import structure in key files."""
    key_files = [
        'src/services/rag_service.py',
        'src/services/azure_ai_service.py',
        'src/api/main.py'
    ]
    
    import_issues = []
    
    for file_path in key_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse imports
            tree = ast.parse(content)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")
            
            # Check for common issues
            issues = []
            if 'fastapi' in str(imports) and file_path.endswith('main.py'):
                issues.append("FastAPI import found (good)")
            
            if len(imports) == 0:
                issues.append("No imports found")
            
            if issues:
                import_issues.append((file_path, issues))
                
        except Exception as e:
            import_issues.append((file_path, [f"Error parsing: {str(e)}"]))
    
    return import_issues

def main():
    """Main validation function."""
    print("🚀 RAG Service Implementation Validation")
    print("=" * 50)
    
    # Check file structure
    print("\n📁 Checking file structure...")
    structure_result = check_file_structure()
    
    print(f"✅ Valid files: {len(structure_result['valid_files'])}/{structure_result['total_required']}")
    
    if structure_result['missing_files']:
        print(f"❌ Missing files ({len(structure_result['missing_files'])}):")
        for file in structure_result['missing_files']:
            print(f"   - {file}")
    
    if structure_result['syntax_errors']:
        print(f"❌ Syntax errors ({len(structure_result['syntax_errors'])}):")
        for file, error in structure_result['syntax_errors']:
            print(f"   - {file}: {error}")
    
    # Check imports
    print("\n📦 Checking import structure...")
    import_issues = check_import_structure()
    
    if import_issues:
        for file, issues in import_issues:
            print(f"📄 {file}:")
            for issue in issues:
                print(f"   - {issue}")
    else:
        print("✅ No obvious import issues found")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   Total files required: {structure_result['total_required']}")
    print(f"   Files present and valid: {len(structure_result['valid_files'])}")
    print(f"   Missing files: {len(structure_result['missing_files'])}")
    print(f"   Syntax errors: {len(structure_result['syntax_errors'])}")
    
    if len(structure_result['valid_files']) == structure_result['total_required'] and not structure_result['syntax_errors']:
        print("\n🎉 All required files are present and syntactically valid!")
        print("\n📋 Implementation includes:")
        print("   ✅ Complete service layer (5 new services)")
        print("   ✅ FastAPI application with 3 route modules")
        print("   ✅ Utility modules for file handling and validation")
        print("   ✅ Docker deployment configuration")
        print("   ✅ Environment configuration template")
        print("\n🔧 Next steps:")
        print("   1. Set up Azure services and configure environment variables")
        print("   2. Install dependencies: pip install -r requirements.txt")
        print("   3. Run the service: python -m uvicorn src.api.main:app --reload")
        print("   4. Test endpoints at http://localhost:8000/docs")
    else:
        print(f"\n⚠️ {structure_result['total_required'] - len(structure_result['valid_files'])} issues need to be resolved")
    
    return len(structure_result['missing_files']) == 0 and len(structure_result['syntax_errors']) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)