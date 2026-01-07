from src.ats_matcher import ATSMatcher

matcher = ATSMatcher()

# Test text resume
score, df = matcher.analyze_resume(
    resume_text="""John Doe - AI/ML Student
Experience: ASL recognition with ResNet on Colab T4. RL with PPO, Gemini chatbot.""",
    jd_path='data/sample_jd.txt'
)

print("\nTop matches:")
print(df.nlargest(3, 'JD Match Score'))
