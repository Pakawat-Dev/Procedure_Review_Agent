import streamlit as st
import os
import tempfile
from docx import Document
from procedure_review_agent import EnhancedQualityReviewAgent
from dotenv import load_dotenv

load_dotenv()

def extract_text_from_docx(uploaded_file):
    """Extract text from uploaded DOCX file"""
    doc = Document(uploaded_file)
    text = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            text.append(paragraph.text)
    return '\n'.join(text)

def main():
    st.set_page_config(
        page_title="Quality Procedure Review Agent",
        page_icon="📋",
        layout="wide"
    )
    
    st.title("📋 Quality Procedure Review Agent")
    st.markdown("Upload a procedure document (DOCX) for ISO compliance review")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        max_iterations = st.slider("Max Review Iterations", 1, 5, 3)
        
        if not api_key:
            st.warning("Please enter your OpenAI API key")
            return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Procedure")
        uploaded_file = st.file_uploader("Choose a DOCX file", type=['docx'])
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Extract and display text
            try:
                procedure_text = extract_text_from_docx(uploaded_file)
                st.subheader("Extracted Text Preview")
                st.text_area("Procedure Content", procedure_text[:1000] + "..." if len(procedure_text) > 1000 else procedure_text, height=200)
                
                # Review button
                if st.button("🔍 Start Review", type="primary"):
                    with st.spinner("Reviewing procedure... This may take a few minutes."):
                        try:
                            # Initialize agent
                            agent = EnhancedQualityReviewAgent(api_key)
                            
                            # Run comprehensive review
                            result = agent.comprehensive_review(procedure_text, max_iterations)
                            
                            # Store result in session state
                            st.session_state.review_result = result
                            st.session_state.procedure_text = procedure_text
                            
                            st.success("Review completed!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error during review: {str(e)}")
                            
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    with col2:
        st.header("Review Results")
        
        if 'review_result' in st.session_state:
            result = st.session_state.review_result
            
            # Display token usage
            st.subheader("🔢 Token Usage")
            if 'token_usage' in result:
                token_col1, token_col2, token_col3 = st.columns(3)
                with token_col1:
                    st.metric("Input Tokens", f"{result['token_usage']['input_tokens']:,}")
                with token_col2:
                    st.metric("Output Tokens", f"{result['token_usage']['output_tokens']:,}")
                with token_col3:
                    st.metric("Total Tokens", f"{result['token_usage']['total_tokens']:,}")
            
            # Display scores
            st.subheader("📊 Evaluation Scores")
            if 'evaluation_result' in result and 'scores' in result['evaluation_result']:
                scores = result['evaluation_result']['scores']
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Completeness", f"{scores.get('completeness', 0)}/5")
                    st.metric("Accuracy", f"{scores.get('accuracy', 0)}/5")
                with col_b:
                    st.metric("Specificity", f"{scores.get('specificity', 0)}/5")
                    st.metric("Risk Assessment", f"{scores.get('risk_assessment', 0)}/5")
                with col_c:
                    st.metric("Documentation", f"{scores.get('documentation', 0)}/5")
                    st.metric("Overall Score", f"{result['evaluation_result'].get('overall_score', 0):.1f}/5")
            
            # Display review findings
            st.subheader("📝 Review Findings")
            st.text_area("Detailed Review", result.get('final_output', ''), height=300)
            
            # Generate and download report
            if st.button("📄 Generate Report", type="secondary"):
                with st.spinner("Generating DOCX report..."):
                    try:
                        # Generate enhanced report
                        agent = EnhancedQualityReviewAgent(api_key)
                        report_filename = agent.generate_enhanced_report(result)
                        
                        # Read the generated file
                        with open(report_filename, 'rb') as file:
                            report_data = file.read()
                        
                        # Provide download button
                        st.download_button(
                            label="⬇️ Download Report",
                            data=report_data,
                            file_name=f"quality_review_report_{result.get('iteration_count', 1)}_iterations.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                        
                        # Clean up temporary file
                        os.remove(report_filename)
                        
                        st.success("Report generated successfully!")
                        
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")
        else:
            st.info("Upload a procedure document and click 'Start Review' to see results here.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Standards Applied:** ISO 13485, ISO 14971, IEC 62304")

if __name__ == "__main__":
    main()