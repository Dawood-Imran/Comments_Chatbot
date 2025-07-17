def handle_greeting_response(question: str) -> str:
    """
    Handle greeting and casual responses without document retrieval
    """
    question_lower = question.lower().strip()
    
    # Different responses based on the type of greeting
    if any(word in question_lower for word in ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']):
        return """# Hello! 👋

I'm your customer feedback analysis assistant. I can help you analyze customer reviews and feedback to provide insights about:

- Customer satisfaction levels
- Common complaints or issues
- Positive feedback patterns
- Specific topics or themes in reviews
- Emotional sentiment analysis
- Individual staff or service feedback

## How can I help you today?

You can ask me questions like:
- "What are customers saying about our service?"
- "Are there any common complaints?"
- "What do customers like most about us?"
- "How satisfied are customers with [specific topic]?"
- "What kind of person is [staff member name]?"

Just type your question and I'll analyze the available customer feedback to give you detailed insights!"""

    elif any(word in question_lower for word in ['help', 'what can you do', 'what can you help']):
        return """# Customer Feedback Analysis Assistant

I can help you analyze customer reviews and feedback data to provide valuable insights for your business.

## What I can analyze:

### 📊 **Satisfaction Analysis**
- Overall customer satisfaction levels
- Positive vs negative feedback patterns
- Customer sentiment trends

### 🔍 **Topic Analysis**
- Specific service or product feedback
- Common themes and issues
- Feature requests and suggestions

### 👥 **Staff Feedback**
- Individual staff member reviews
- Service quality assessments
- Professional reputation analysis

### 😊 **Emotional Insights**
- Customer emotion breakdown
- Sentiment analysis
- Impact assessment

## How to ask questions:

Simply type your question in natural language, such as:
- "What are the main customer complaints?"
- "How do customers feel about our service?"
- "What feedback do we have about Dr. Smith?"
- "Are customers satisfied with wait times?"

I'll analyze the available reviews and provide detailed insights with actual customer quotes and emotion analysis!"""

    elif any(word in question_lower for word in ['thank you', 'thanks']):
        return """# You're welcome! 😊

I'm always here to help you analyze customer feedback and provide insights. If you have any other questions about your customer reviews or need analysis on specific topics, just let me know!

## Need more help?
Feel free to ask me about:
- Customer satisfaction trends
- Specific feedback topics
- Staff performance reviews
- Service quality analysis

What would you like to explore next?"""

    elif any(word in question_lower for word in ['bye', 'goodbye', 'see you']):
        return """# Goodbye! 👋

Thank you for using the customer feedback analysis tool. I hope the insights were helpful for your business!

Feel free to come back anytime you need to analyze customer reviews or feedback. Have a great day!"""

    else:
        return """# Hi there! 👋

I'm your customer feedback analysis assistant. I can help you understand what customers are saying about your business by analyzing reviews and feedback.

## What would you like to know?

You can ask me questions like:
- "What are customers saying about our service?"
- "Are there any common issues?"
- "How satisfied are customers?"
- "What feedback do we have about [specific topic]?"

Just type your question and I'll provide detailed insights based on the available customer feedback!"""
