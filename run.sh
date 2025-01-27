# run the example model after environment setup

# chose one of the three llm supplier

# LLM=deepseek/deepseek-chat
# nohup python -m examples.gpt4o \
#         --llm  $LLM\
#         --middle_out \
#         --supplier openrouter \
#         --retriever searxng \
#         --topic "composition of human breast milk" \
#         --depth 3 \
#         --outputdir results/$(basename $LLM) \
#         > log-$(basename $LLM).txt &     


# LLM=deepseek-chat
# nohup python -m examples.gpt4o \
#         --llm  $LLM\
#         --supplier DMX \
#         --retriever searxng \
#         --topic "composition of human breast milk" \
#         --depth 3 \
#         --outputdir results/$(basename $LLM) \
#         > log-$(basename $LLM).txt &     


LLM=deepseek-chat
nohup python -m examples.gpt4o \
        --llm  $LLM\
        --supplier deepseek \
        --retriever searxng \
        --topic "composition of human breast milk" \
        --depth 3 \
        --outputdir results/$(basename $LLM) \
        > log-$(basename $LLM).txt &     