function useModel(sentence1, sentence2, callback) {
    use.load().then((model) => {
      embedSentences(model, sentence1, sentence2, callback);
    });
  }
  
  function embedSentences(model, sentence1, sentence2, callback) {
    const sentences = [sentence1, sentence2];
    model.embed(sentences).then((embeddings) => {
      const embeds = embeddings.arraySync();
      const sentence1Embedding = embeds[0];
      const sentence2Embedding = embeds[1];
      getSimilarityPercent(sentence1Embedding, sentence2Embedding, callback);
    });
  }
  
  function getSimilarityPercent(embed1, embed2, callback) {
    const similarity = cosineSimilarity(embed1, embed2);
    if (callback) callback(similarity);
    return similarity;
  }
  
  function cosineSimilarity(a, b) {  
    const magnitudeA = Math.sqrt(dotProduct(a, a));
    const magnitudeB = Math.sqrt(dotProduct(b, b));
    if (magnitudeA && magnitudeB) {
      return dotProduct(a, b) / (magnitudeA * magnitudeB);
    } else {
      return 0;
    }
  }
  
  function dotProduct(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += a[i] * b[i];
    }
    return sum;
  }
  
  // -------------------------------------------------
  
  const runFromCli = typeof require !== "undefined" && require.main === module;
  if (runFromCli) {
    useModelToEmbedAllSentences(["cat", "dog"]);
  }
  
  function useModelToEmbedAllSentences(sentences, callback) {
    require("@tensorflow/tfjs-node");
    const use = require("@tensorflow-models/universal-sentence-encoder");
    const fs = require("fs");
    use.load().then((model) => {
      embedAllSentences(model, sentences, fs);
    });
  }
  
  function embedAllSentences(model, sentences, fs) {
    model.embed(sentences).then((embeddings) => {
      const embeds = embeddings.arraySync();
      if (fs) {
        for (let i = 0; i < embeds.length; i++) {
          const sentence = sentences[i];
          const embed = embeds[i];
          const addNewLine = i === 0 ? "" : "\n";
          fs.appendFile("words.txt", addNewLine + sentence, function (err) {
            if (err) throw err;
            console.log(`Added word ${i}!`);
          });
          fs.appendFile("embeddings.txt", addNewLine + embed, function (err) {
            if (err) throw err;
            console.log(`Added embedding ${i}!`);
          });
        }
        console.log("Done adding all words and embeddings (mapped by index).");
      }
    });
  }