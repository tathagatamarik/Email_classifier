"""
Email Classifier Streamlit App
==============================
A professional web interface for classifying customer inquiry emails 
and generating automated quotations.

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime
from io import BytesIO

# Import classifiers
from email_classifier import EmailClassifier
from llm_email_classifier import MockLLMClassifier, LLMEmailClassifier

# Page configuration
st.set_page_config(
    page_title="JWI Email Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .quotation-box {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 0.5rem;
        border: 2px solid #dee2e6;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_erp_database(excel_file):
    """Load ERP database from Excel file"""
    try:
        df = pd.read_excel(excel_file, sheet_name='Materials')
        df = df.drop_duplicates(subset=['Material_Code'], keep='first')
        
        # Auto-detect price column (Unit_Price or Unit_Price_USD)
        price_column = None
        if 'Unit_Price_USD' in df.columns:
            price_column = 'Unit_Price_USD'
        elif 'Unit_Price' in df.columns:
            price_column = 'Unit_Price'
        else:
            raise ValueError("No price column found. Expected 'Unit_Price' or 'Unit_Price_USD'")
        
        material_codes = set(df['Material_Code'].values)
        price_dict = dict(zip(df['Material_Code'], df[price_column]))
        material_info = df.set_index('Material_Code').to_dict('index')
        
        return material_codes, price_dict, material_info, df
    except Exception as e:
        st.error(f"Error loading ERP database: {e}")
        return set(), {}, {}, pd.DataFrame()


@st.cache_resource
def initialize_classifiers(_material_codes):
    """Initialize both classifiers"""
    regex_classifier = EmailClassifier(erp_material_db=_material_codes)
    mock_llm_classifier = MockLLMClassifier(erp_material_db=_material_codes)
    return regex_classifier, mock_llm_classifier


def initialize_real_llm_classifier(provider, model, api_key, material_codes):
    """Initialize real LLM classifier"""
    try:
        if provider == "OpenAI":
            os.environ['OPENAI_API_KEY'] = api_key
        else:  # Anthropic
            os.environ['ANTHROPIC_API_KEY'] = api_key
        
        classifier = LLMEmailClassifier(
            provider=provider.lower(),
            model=model,
            erp_material_db=material_codes,
            temperature=0.1
        )
        return classifier, None
    except Exception as e:
        return None, str(e)


def generate_quotation_data(materials, price_dict, material_info, quantities=None):
    """Generate quotation line items"""
    quotation_items = []
    total_amount = 0.0
    
    for i, material in enumerate(materials):
        mat_code = material.material_number if hasattr(material, 'material_number') else material
        mat_code_normalized = str(mat_code).replace("-", "").replace(" ", "").upper()
        
        if mat_code_normalized in price_dict:
            price = price_dict[mat_code_normalized]
            info = material_info.get(mat_code_normalized, {})
            
            # Use provided quantity or default to 100
            quantity = quantities[i] if quantities and i < len(quantities) else 100
            
            item = {
                'Material_Code': mat_code,  # Use original code for display
                'Description': info.get('Description', 'N/A'),
                'Unit_Price': price,
                'Quantity': quantity,
                'Line_Total': price * quantity,
                'Lead_Time_Days': info.get('Lead_Time_Days', 'TBD'),
                'Stock_Status': info.get('Stock_Status', 'Check'),
                'Supplier': info.get('Supplier', 'N/A')
            }
            
            quotation_items.append(item)
            total_amount += item['Line_Total']
    
    return quotation_items, total_amount


def format_quotation_html(quotation_items, total_amount, customer_email="", quote_number=None, currency="USD"):
    """Generate HTML formatted quotation"""
    if quote_number is None:
        quote_number = f"Q-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    valid_until = datetime.now().strftime('%Y-%m-%d')
    
    # Currency symbols
    currency_symbols = {
        'USD': '$',
        'INR': '₹',
        'EUR': '€',
        'GBP': '£'
    }
    symbol = currency_symbols.get(currency, currency + ' ')
    
    html = f"""
    <div class="quotation-box">
        <h2 style="text-align: center; color: #1f77b4; margin-bottom: 1.5rem;">
            QUOTATION
        </h2>
        
        <div style="display: flex; justify-content: space-between; margin-bottom: 2rem;">
            <div>
                <strong>JWI Manufacturing Solutions</strong><br>
                Email: sales@jwi.com<br>
                Phone: +1-555-0100<br>
                Web: www.jwi.com
            </div>
            <div style="text-align: right;">
                <strong>Quote #:</strong> {quote_number}<br>
                <strong>Date:</strong> {current_date}<br>
                <strong>Valid Until:</strong> {valid_until} (30 days)
            </div>
        </div>
        
        <hr style="border: 1px solid #dee2e6; margin: 1.5rem 0;">
        
        <table style="width: 100%; border-collapse: collapse; margin: 1rem 0;">
            <thead>
                <tr style="background-color: #e9ecef; border-bottom: 2px solid #dee2e6;">
                    <th style="padding: 0.75rem; text-align: left;">Item</th>
                    <th style="padding: 0.75rem; text-align: left;">Description</th>
                    <th style="padding: 0.75rem; text-align: center;">Qty</th>
                    <th style="padding: 0.75rem; text-align: right;">Unit Price</th>
                    <th style="padding: 0.75rem; text-align: right;">Total</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for item in quotation_items:
        desc = item['Description'][:50] + '...' if len(item['Description']) > 50 else item['Description']
        html += f"""
                <tr style="border-bottom: 1px solid #dee2e6;">
                    <td style="padding: 0.75rem;">{item['Material_Code']}</td>
                    <td style="padding: 0.75rem;">{desc}</td>
                    <td style="padding: 0.75rem; text-align: center;">{item['Quantity']}</td>
                    <td style="padding: 0.75rem; text-align: right;">{symbol}{item['Unit_Price']:.2f}</td>
                    <td style="padding: 0.75rem; text-align: right;">{symbol}{item['Line_Total']:.2f}</td>
                </tr>
                <tr>
                    <td colspan="5" style="padding: 0.25rem 0.75rem; font-size: 0.85rem; color: #6c757d;">
                        Lead Time: {item['Lead_Time_Days']} days | Stock: {item['Stock_Status']} | 
                        Supplier: {item['Supplier']}
                    </td>
                </tr>
        """
    
    html += f"""
            </tbody>
        </table>
        
        <hr style="border: 1px solid #dee2e6; margin: 1.5rem 0;">
        
        <div style="text-align: right; font-size: 1.2rem; margin: 1rem 0;">
            <strong>TOTAL: {symbol}{total_amount:,.2f}</strong>
        </div>
        
        <hr style="border: 1px solid #dee2e6; margin: 1.5rem 0;">
        
        <div style="margin-top: 2rem; font-size: 0.9rem;">
            <strong>Terms & Conditions:</strong>
            <ul>
                <li>Payment: Net 30 days</li>
                <li>Prices in {currency}, FOB shipping point</li>
                <li>Lead times subject to material availability</li>
                <li>Minimum order value: {symbol}100.00</li>
                <li>Prices valid for 30 days from quote date</li>
            </ul>
        </div>
        
        <div style="margin-top: 2rem; padding: 1rem; background-color: #e7f3ff; border-radius: 0.25rem;">
            <strong>📧 Next Steps:</strong><br>
            To proceed with this order, please reply to this quotation with your purchase order 
            or contact our sales team at sales@jwi.com or +1-555-0100.
        </div>
    </div>
    """
    
    return html


def main():
    # Header
    st.markdown('<h1 class="main-header">📧 JWI Email Classifier & Quotation Generator</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")
    
    # ERP Database
    st.sidebar.subheader("ERP Database")
    erp_file = st.sidebar.text_input(
        "ERP Excel File Path",
        value="sample_erp_database.xlsx",
        help="Path to the ERP database Excel file"
    )
    
    # Classifier Selection
    st.sidebar.subheader("Classifier Settings")
    classifier_type = st.sidebar.radio(
        "Select Classifier",
        ["Regex (Fast & Free)", "Mock LLM (Test)", "Real LLM (OpenAI/Anthropic)"],
        help="Choose the classification method"
    )
    
    # Real LLM Configuration
    use_real_llm = False
    llm_classifier = None
    
    if classifier_type == "Real LLM (OpenAI/Anthropic)":
        use_real_llm = True
        st.sidebar.subheader("LLM API Configuration")
        
        llm_provider = st.sidebar.selectbox(
            "Provider",
            ["OpenAI", "Anthropic"]
        )
        
        if llm_provider == "OpenAI":
            llm_model = st.sidebar.selectbox(
                "Model",
                ["gpt-4-turbo-preview", "gpt-3.5-turbo", "gpt-4"]
            )
        else:
            llm_model = st.sidebar.selectbox(
                "Model",
                ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307", "claude-3-opus-20240229"]
            )
        
        api_key = st.sidebar.text_input(
            "API Key",
            type="password",
            help="Enter your API key"
        )
    
    # Load ERP Database
    if os.path.exists(erp_file):
        material_codes, price_dict, material_info, df_erp = load_erp_database(erp_file)
        
        if len(material_codes) > 0:
            st.sidebar.success(f"✓ Loaded {len(material_codes)} materials")
            
            # Detect currency from dataframe
            currency = 'USD'
            price_col_name = 'Unit_Price_USD'
            if 'Currency' in df_erp.columns and not df_erp['Currency'].isna().all():
                currency = df_erp['Currency'].iloc[0]
            if 'Unit_Price' in df_erp.columns:
                price_col_name = 'Unit_Price'
                if currency == 'USD':
                    currency = df_erp.get('Currency', pd.Series(['INR'])).iloc[0] if 'Currency' in df_erp.columns else 'INR'
            
            currency_symbols = {'USD': '$', 'INR': '₹', 'EUR': '€', 'GBP': '£'}
            symbol = currency_symbols.get(currency, currency + ' ')
            
            total_value = df_erp[price_col_name].sum()
            st.sidebar.metric("Total ERP Value", f"{symbol}{total_value:,.2f}")
        else:
            st.sidebar.error("❌ Failed to load ERP database")
            return
    else:
        st.sidebar.error(f"❌ ERP file not found: {erp_file}")
        st.info("💡 Please run `python generate_erp_database.py` first to create the sample database.")
        return
    
    # Initialize Classifiers
    regex_classifier, mock_llm_classifier = initialize_classifiers(material_codes)
    
    if use_real_llm and api_key:
        with st.spinner("Initializing LLM classifier..."):
            llm_classifier, error = initialize_real_llm_classifier(
                llm_provider, llm_model, api_key, material_codes
            )
            if error:
                st.sidebar.error(f"❌ LLM initialization failed: {error}")
                use_real_llm = False
            else:
                st.sidebar.success("✓ LLM classifier ready")
    
    # Main Content Area
    st.header("📩 Email Input")
    
    # Sample emails
    sample_emails = {
        "Custom (Enter your own)": "",
        "Standard Inquiry - All Valid Materials": """From: john.smith@manufacturing.com
To: sales@jwi.com
Subject: Quote Request - Production Line Components

Hi JWI Team,

We need a quote for the following items:

1. JWI-12345678 - Qty: 100 units
2. MAT-456789 - Qty: 250 units  
3. PN-987654 - Qty: 150 units
4. AB-123456 - Qty: 500 units

Please provide your best pricing and lead times.

Regards,
John Smith
ABC Manufacturing""",
        
        "Mixed Inquiry - Standard + Custom": """From: sarah.jones@engineering.com
To: sales@jwi.com
Subject: Standard Parts + Custom Design

Hello,

STANDARD ITEMS:
- JWI-12345678 x 50 units
- HYD-555666 x 25 units
- COMP-001234 x 200 units

CUSTOM REQUIREMENT:
We also need a custom hydraulic manifold:
- Operating pressure: 350 bar
- 6 stations with individual flow control
- Aluminum body with hard anodizing
- Qty: 30 units

Can you quote both separately?

Thanks,
Sarah Jones""",
        
        "Non-Standard Inquiry - Custom Only": """From: mike.wilson@oilgas.com
To: sales@jwi.com
Subject: Custom Hydraulic Valve Design

Hello JWI Engineering,

We require a custom hydraulic valve for offshore drilling:

SPECIFICATIONS:
- Operating pressure: 5000 PSI continuous
- Port configuration: 4-way, 3-position
- Material: Stainless Steel 316L
- Flow rate: 20 GPM at 3000 PSI
- Temperature: -20°C to 150°C
- Certifications: API 6A, NACE MR0175

Quantity: 50 units initially

Please provide feasibility and pricing.

Best regards,
Mike Wilson"""
    }
    
    selected_sample = st.selectbox(
        "Select Sample Email (or choose 'Custom' to enter your own)",
        list(sample_emails.keys())
    )
    
    email_text = st.text_area(
        "Email Content",
        value=sample_emails[selected_sample],
        height=300,
        help="Paste or type the customer inquiry email here"
    )
    
    # Classification Button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        classify_button = st.button("🔍 Classify Email & Generate Quotation", 
                                    type="primary", use_container_width=True)
    
    # Process Email
    if classify_button and email_text:
        with st.spinner("Processing email..."):
            # Select classifier
            if use_real_llm and llm_classifier:
                classifier = llm_classifier
                classifier_name = f"{llm_provider} - {llm_model}"
            elif classifier_type == "Mock LLM (Test)":
                classifier = mock_llm_classifier
                classifier_name = "Mock LLM"
            else:
                classifier = regex_classifier
                classifier_name = "Regex Classifier"
            
            # Classify
            try:
                result = classifier.classify(email_text)
                
                # Display Results
                st.header("📊 Classification Results")
                
                # Metrics Row
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    inquiry_type = result.inquiry_type.value if hasattr(result, 'inquiry_type') else result.classification
                    type_color = {
                        'standard': '🟢',
                        'non_standard': '🔴',
                        'mixed': '🟡'
                    }.get(inquiry_type.lower(), '⚪')
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{type_color} Inquiry Type</h3>
                        <h2>{inquiry_type.upper()}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    confidence = result.overall_confidence if hasattr(result, 'overall_confidence') else result.confidence_score
                    confidence_pct = confidence * 100 if confidence <= 1 else confidence
                    
                    confidence_color = '🟢' if confidence_pct >= 85 else '🟡' if confidence_pct >= 60 else '🔴'
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{confidence_color} Confidence Score</h3>
                        <h2>{confidence_pct:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    materials_found = len(result.standard_materials) if hasattr(result, 'standard_materials') else 0
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📦 Materials Found</h3>
                        <h2>{materials_found}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Detailed Results
                col_left, col_right = st.columns([1, 1])
                
                with col_left:
                    st.subheader("📋 Extracted Materials")
                    
                    if result.standard_materials:
                        materials_data = []
                        for i, mat in enumerate(result.standard_materials, 1):
                            mat_code = mat.material_number if hasattr(mat, 'material_number') else mat
                            mat_confidence = mat.confidence if hasattr(mat, 'confidence') else 1.0
                            
                            # Normalize material code (remove hyphens, spaces)
                            mat_code_normalized = str(mat_code).replace("-", "").replace(" ", "").upper()
                            
                            # Check if in ERP
                            in_erp = mat_code_normalized in material_codes
                            status = "✓ Valid" if in_erp else "✗ Not Found"
                            
                            materials_data.append({
                                '#': i,
                                'Material': mat_code,  # Show original code
                                'Status': status,
                                'Confidence': f"{mat_confidence*100:.0f}%" if mat_confidence <= 1 else f"{mat_confidence:.0f}%"
                            })
                        
                        df_materials = pd.DataFrame(materials_data)
                        st.dataframe(df_materials, use_container_width=True, hide_index=True)
                        
                        # ERP Validation Summary
                        valid_count = sum(1 for mat in result.standard_materials 
                                        if str((mat.material_number if hasattr(mat, 'material_number') else mat)).replace("-", "").replace(" ", "").upper() in material_codes)
                        invalid_count = len(result.standard_materials) - valid_count
                        
                        if valid_count == len(result.standard_materials):
                            st.markdown(f"""
                            <div class="success-box">
                                <strong>✓ All materials validated in ERP</strong><br>
                                {valid_count} valid materials ready for quotation
                            </div>
                            """, unsafe_allow_html=True)
                        elif valid_count > 0:
                            st.markdown(f"""
                            <div class="warning-box">
                                <strong>⚠ Partial validation</strong><br>
                                {valid_count} valid, {invalid_count} not found in ERP
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="error-box">
                                <strong>✗ No valid materials found</strong><br>
                                All materials need manual review
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No standard materials found in this email.")
                
                with col_right:
                    st.subheader("📝 Classification Details")
                    
                    st.write(f"**Classifier Used:** {classifier_name}")
                    
                    if hasattr(result, 'reasoning'):
                        st.write("**Reasoning:**")
                        st.info(result.reasoning)
                    
                    # Non-standard sections
                    if hasattr(result, 'non_standard_sections') and result.non_standard_sections:
                        st.write("**Non-Standard Sections:**")
                        for i, section in enumerate(result.non_standard_sections, 1):
                            with st.expander(f"Section {i}"):
                                st.write(section)
                    elif hasattr(result, 'non_standard_text') and result.non_standard_text:
                        st.write("**Non-Standard Content:**")
                        for i, text in enumerate(result.non_standard_text, 1):
                            with st.expander(f"Section {i}"):
                                st.write(text)
                
                # Generate Quotation
                st.markdown("---")
                st.header("💰 Quotation")
                
                if result.standard_materials:
                    # Filter for valid materials using normalized codes
                    valid_materials = [mat for mat in result.standard_materials 
                                      if str((mat.material_number if hasattr(mat, 'material_number') else mat)).replace("-", "").replace(" ", "").upper() in material_codes]
                    
                    if valid_materials:
                        # Get quantities (parse from email or use defaults)
                        quotation_items, total_amount = generate_quotation_data(
                            valid_materials, price_dict, material_info
                        )
                        
                        if quotation_items:
                            # Get currency from ERP dataframe
                            currency = df_erp.get('Currency', pd.Series(['USD'])).iloc[0] if 'Currency' in df_erp.columns else 'USD'
                            
                            # Display quotation
                            quotation_html = format_quotation_html(quotation_items, total_amount, currency=currency)
                            st.markdown(quotation_html, unsafe_allow_html=True)
                            
                            # Download buttons
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                # CSV download
                                df_quote = pd.DataFrame(quotation_items)
                                csv = df_quote.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download CSV",
                                    data=csv,
                                    file_name=f"quotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                            
                            with col2:
                                # HTML download
                                st.download_button(
                                    label="📄 Download HTML",
                                    data=quotation_html,
                                    file_name=f"quotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                    mime="text/html"
                                )
                            
                            # Action recommendations
                            st.markdown("---")
                            st.subheader("📋 Recommended Actions")
                            
                            if inquiry_type.lower() == 'standard':
                                st.markdown("""
                                <div class="success-box">
                                    <strong>✓ Fully Automated Processing</strong>
                                    <ul>
                                        <li>All materials found in ERP</li>
                                        <li>Quotation generated automatically</li>
                                        <li>Ready to send to customer</li>
                                        <li>Estimated response time: < 2 minutes</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                            elif inquiry_type.lower() == 'mixed':
                                st.markdown("""
                                <div class="warning-box">
                                    <strong>⚠ Hybrid Processing Required</strong>
                                    <ul>
                                        <li>Standard materials: Automated quotation (see above)</li>
                                        <li>Custom sections: Require engineering review</li>
                                        <li>Create ticket for custom parts</li>
                                        <li>Send combined response with standard quote + custom timeline</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div class="error-box">
                                    <strong>⚠ Manual Review Required</strong>
                                    <ul>
                                        <li>Route to sales/engineering team</li>
                                        <li>Assess custom requirements</li>
                                        <li>Prepare custom solution quotation</li>
                                        <li>Schedule technical review call if needed</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                        
                    else:
                        st.warning("No valid materials found in ERP. Manual review required.")
                else:
                    st.info("This appears to be a non-standard inquiry requiring custom engineering review.")
            
            except Exception as e:
                st.error(f"Error during classification: {e}")
                st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; padding: 2rem;">
        <p>JWI Email Classifier & Quotation Generator v1.0</p>
        <p>Powered by Advanced AI Classification Technology</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()