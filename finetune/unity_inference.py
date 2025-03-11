from unity_train import inference

if __name__ == "__main__":    
    # Test inference with some examples
    test_prompts = [
        "How do I add force to a rigidbody in Unity?",
        "My game crashes when I try to load a new scene. Can you help me debug?",
        "What's the recipe for beef stroganoff?",
        "Where is Seattle located?"
    ]
    
    
    for prompt in test_prompts:
        classification, result = inference(prompt)
        print(f"Prompt: {prompt}")
        print(f"Classification: {classification}")
        #print(f'result={result}')
        print("----")        