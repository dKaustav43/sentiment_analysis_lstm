 # #Inference function/layer.


    # def predict_sentiment(
    #         text:str,
    #         vocab=vocab,
    #         max_length=300,
    #         device=device
    # ):
    #     classifier_model.eval()
    #     #tokenize
    #     tokens = text.lower().split()

    #     #encode
    #     encoded = encode_tokens(tokens, vocab)
    #     encoded = truncation(encoded, max_length)
    #     encoded = pad_sequence(encoded, max_length)


    #     x = torch.tensor(encoded).unsqueeze(0).to(device)

    #     with torch.no_grad():
    #         logit = classifier_model(x).squeeze()
    #         prob = torch.sigmoid(logit).item()

    #     label = "positive" if prob >= 0.5 else "negetive"

    #     return label, prob


    # label,prob = predict_sentiment(
    #     "The movie was was well done!"
    # )
    # print(label,prob)