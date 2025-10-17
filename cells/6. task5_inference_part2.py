# ============================================================================
# CELL 6 - PART 2: TASK 5 - INFERENCE & UI (Gradio Interface)
# ============================================================================
# Empathetic Conversational Chatbot - Gradio UI
# 
# This is PART 2 of Task 5, containing:
# - Gradio chatbot interface
# - Conversation history tracking
# - Decoding strategy selection
# - Interactive demo
# 
# NOTE: Run PART 1 first before running this cell!
# ============================================================================

# ----------------------------------------------------------------------------
# STEP 1: Install and Import Gradio
# ----------------------------------------------------------------------------
print("=" * 80)
print("TASK 5 - PART 2: SETTING UP GRADIO UI")
print("=" * 80)

!pip install gradio -q

import gradio as gr
import datetime

print("✓ Gradio installed and imported!")
print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 2: Conversation History Manager
# ----------------------------------------------------------------------------
print("=" * 80)
print("SETTING UP CONVERSATION MANAGER")
print("=" * 80)

class ConversationHistory:
    """Manage conversation history."""
    
    def __init__(self):
        self.history = []
    
    def add_exchange(self, customer, agent, emotion, situation):
        """Add a customer-agent exchange to history."""
        self.history.append({
            'timestamp': datetime.datetime.now().strftime("%H:%M:%S"),
            'customer': customer,
            'agent': agent,
            'emotion': emotion,
            'situation': situation
        })
    
    def get_formatted_history(self):
        """Get formatted conversation history."""
        if not self.history:
            return "No conversation history yet."
        
        formatted = []
        for i, exchange in enumerate(self.history, 1):
            formatted.append(f"**Turn {i}** ({exchange['timestamp']}) - Emotion: *{exchange['emotion']}*")
            formatted.append(f"👤 **Customer:** {exchange['customer']}")
            formatted.append(f"🤖 **Agent:** {exchange['agent']}")
            formatted.append("---")
        
        return "\n".join(formatted)
    
    def clear(self):
        """Clear conversation history."""
        self.history = []


# Initialize conversation manager
conversation_manager = ConversationHistory()

print("✓ Conversation manager initialized")
print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 3: Gradio Interface Functions
# ----------------------------------------------------------------------------
print("=" * 80)
print("DEFINING GRADIO INTERFACE FUNCTIONS")
print("=" * 80)

def chatbot_interface(customer_message, emotion, situation, decoding_strategy, beam_width, conversation_history):
    """
    Main chatbot interface function for Gradio.
    
    Args:
        customer_message: Customer's input message
        emotion: Selected emotion
        situation: Situation description
        decoding_strategy: 'Greedy' or 'Beam Search'
        beam_width: Beam width for beam search
        conversation_history: Current conversation history HTML
    
    Returns:
        Tuple of (agent_response, updated_conversation_history)
    """
    if not customer_message.strip():
        return "Please enter a message.", conversation_history
    
    # Map strategy name to function parameter
    strategy_map = {
        'Greedy': 'greedy',
        'Beam Search': 'beam_search'
    }
    
    # Generate response
    result = generate_response(
        emotion=emotion,
        situation=situation,
        customer_utterance=customer_message,
        decoding_strategy=strategy_map[decoding_strategy],
        beam_width=int(beam_width)
    )
    
    agent_response = result['response']
    
    # Add to conversation history
    conversation_manager.add_exchange(
        customer=customer_message,
        agent=agent_response,
        emotion=emotion,
        situation=situation
    )
    
    # Get updated history
    updated_history = conversation_manager.get_formatted_history()
    
    return agent_response, updated_history


def clear_conversation():
    """Clear conversation history."""
    conversation_manager.clear()
    return "", "Conversation history cleared."


print("✓ Gradio functions defined")
print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 4: Create Gradio Interface
# ----------------------------------------------------------------------------
print("=" * 80)
print("CREATING GRADIO INTERFACE")
print("=" * 80)

# Define emotion options
emotion_options = [
    "happy", "sad", "angry", "anxious", "excited", "grateful", 
    "surprised", "disappointed", "proud", "afraid", "annoyed",
    "caring", "confident", "content", "disgusted", "embarrassed",
    "faithful", "furious", "guilty", "hopeful", "impressed",
    "jealous", "joyful", "lonely", "nostalgic", "prepared",
    "sentimental", "terrified", "trusting", "devastated", "anticipating",
    "apprehensive", "ashamed", "neutral"
]

# Create Gradio interface with Blocks for more control
with gr.Blocks(title="Empathetic Chatbot", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown(
        """
        # 🤖 Empathetic Conversational Chatbot
        
        **Built with Transformer (Multi-Head Attention) from Scratch**
        
        This chatbot generates empathetic responses based on:
        - Your message
        - The emotional context
        - The situation you're in
        
        ---
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat Interface")
            
            # Input fields
            customer_input = gr.Textbox(
                label="Your Message",
                placeholder="Type your message here...",
                lines=3
            )
            
            with gr.Row():
                emotion_input = gr.Dropdown(
                    choices=emotion_options,
                    label="Emotion",
                    value="happy",
                    info="Select the emotion that best describes the context"
                )
                
                situation_input = gr.Textbox(
                    label="Situation (Optional)",
                    placeholder="Describe the situation briefly...",
                    lines=2
                )
            
            # Decoding options
            gr.Markdown("### ⚙️ Decoding Settings")
            
            with gr.Row():
                decoding_strategy = gr.Radio(
                    choices=["Greedy", "Beam Search"],
                    label="Decoding Strategy",
                    value="Greedy",
                    info="Greedy: Fast, Beam Search: Better quality"
                )
                
                beam_width = gr.Slider(
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=5,
                    label="Beam Width",
                    info="Only used for Beam Search"
                )
            
            # Buttons
            with gr.Row():
                send_button = gr.Button("Send Message", variant="primary")
                clear_button = gr.Button("Clear History", variant="secondary")
            
            # Response output
            agent_response = gr.Textbox(
                label="🤖 Agent Response",
                lines=3,
                interactive=False
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### 📜 Conversation History")
            
            conversation_display = gr.Markdown(
                value="No conversation history yet.",
                label="History"
            )
    
    # Examples
    gr.Markdown("### 💡 Try These Examples")
    
    gr.Examples(
        examples=[
            ["I just got a promotion at work!", "excited", "I've been working really hard for this", "Greedy", 5],
            ["I'm feeling really down today.", "sad", "Things haven't been going well lately", "Beam Search", 5],
            ["I can't believe this happened to me!", "surprised", "This was totally unexpected", "Greedy", 5],
            ["I'm worried about my exam tomorrow.", "anxious", "I haven't prepared as much as I should have", "Beam Search", 3],
            ["Thank you so much for your help!", "grateful", "You really made my day", "Greedy", 5],
        ],
        inputs=[customer_input, emotion_input, situation_input, decoding_strategy, beam_width],
        label="Click an example to try it"
    )
    
    # Footer
    gr.Markdown(
        """
        ---
        **Model Details:**
        - Architecture: Transformer Encoder-Decoder (built from scratch)
        - Training: 51,708 empathetic dialogues
        - Metrics: BLEU 97.85, ROUGE-L 0.15, chrF 74.04
        
        **Decoding Strategies:**
        - **Greedy**: Selects most probable token at each step (faster)
        - **Beam Search**: Explores multiple paths (better quality, slower)
        """
    )
    
    # Event handlers
    send_button.click(
        fn=chatbot_interface,
        inputs=[customer_input, emotion_input, situation_input, decoding_strategy, beam_width, conversation_display],
        outputs=[agent_response, conversation_display]
    )
    
    clear_button.click(
        fn=clear_conversation,
        inputs=[],
        outputs=[customer_input, conversation_display]
    )

print("✓ Gradio interface created!")
print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 5: Launch Gradio Interface
# ----------------------------------------------------------------------------
print("=" * 80)
print("LAUNCHING GRADIO INTERFACE")
print("=" * 80)

# Launch the interface
demo.launch(
    share=True,  # Create public link
    debug=False,
    show_error=True
)

print("\n" + "=" * 80)
print("✅ TASK 5 COMPLETE - GRADIO UI LAUNCHED!")
print("=" * 80)
print("\n🌐 The interface is now running!")
print("📱 A public link has been generated for sharing")
print("=" * 80)

