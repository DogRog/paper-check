# Paper Check

Paper Check is an AI-powered tool designed to assist researchers and students in reviewing and improving their academic papers. It leverages Large Language Models (LLMs) to analyze PDF documents, providing feedback on tone, structure, coherence, and citations. The system features a web-based interface for easy interaction and supports various LLM backends. It also includes a custom finetuned Qwen model for specialized scoring.

## Features

- **PDF Analysis**: Upload and analyze academic papers in PDF format.
- **Multi-Model Support**: Choose from various LLMs like Gemini 2.5 Flash, Qwen 2.5 72B, DeepSeek R1, and Grok 4.1 (you can add more by modifying dropdown in `frontend/index.html`).
- **Custom Scoring Agents**: Option to use a base or finetuned Qwen 3 14B model for scoring.
- **Targeted Feedback**: Filter feedback based on specific criteria:
  - Tone
  - Structure
  - Coherence
  - Citation
- **Interactive UI**: View annotations and feedback directly on the PDF.

## Project Structure

- `backend/`: FastAPI application handling API requests, PDF processing, and LLM interaction.
- `frontend/`: Static HTML/JS/CSS frontend for the user interface.
- `model/`: Scripts and data for finetuning the Qwen model using Unsloth and the PeerRead dataset.

## Architecture

The system uses a graph-based agent workflow powered by LangGraph to analyze papers. Multiple specialized agents run in parallel to evaluate different aspects of the paper, followed by a coordinator that synthesizes the findings and a scoring agent that provides a final assessment.

[![](https://mermaid.ink/img/pako:eNqVU8tq4zAU_RWhUOiAYmzLcWMXBgYnywkFd1WnBFW-jk0VycgybSfk30d-NpndeCF0ju8996kz5ioHHOOjZnWJnjd7iex3d4d2lm8GlN5nh0NjmDaHw-uPgZvtnktAAXpimgkB4tpt52VGSXgdkZ81RrfctHqmaMZVCRokn6kg45VhplLSMgOXWCul80oyo_Rol_Ks4UrDhLddiiDzIcE5uURJCbxTa2bOc1Da1YKaWlSmQZU0ylZQM1NO9aLl8qdN_wb5N4jeoOAqpO-gX0JMgugE-gjojfF3ZMMk34VMPeoVkqlHN4jeoGBCcyjq9N3X0BikCmTsvRDqY-xab57yqV893F55p-ZLVPKI7p9aXQvopE4wTpcL1jQbKFDd_-uGiopKiHhR0MItCmJHqd4hXqyZ_wb-CJcfVW7K2K8_CVdC6XhBKX38R7Hfo63MRz1wc7dg_6d3pWgbSHY-2VGyC0hCbJnfGV9HRinZzqEfMbH7XuU4tgsJBNsZnVgH8blz2WPTtWKPY3vNoWCtMHu8lxfrVjP5otRp8tSqPZY4LphoLGrrnBnYVMw-ptPM2vXOQSeqlQbHUeD1Ijg-408ce17gPIRRFIVhQF17EPxlWRo5dL3yVzRyQ391IfhPH9N1wnDtP3jRirrrgAauTzDklV2m38Mj7t_y5S-eWySY?type=png)](https://mermaid.live/edit#pako:eNqVU8tq4zAU_RWhUOiAYmzLcWMXBgYnywkFd1WnBFW-jk0VycgybSfk30d-NpndeCF0ju8996kz5ioHHOOjZnWJnjd7iex3d4d2lm8GlN5nh0NjmDaHw-uPgZvtnktAAXpimgkB4tpt52VGSXgdkZ81RrfctHqmaMZVCRokn6kg45VhplLSMgOXWCul80oyo_Rol_Ks4UrDhLddiiDzIcE5uURJCbxTa2bOc1Da1YKaWlSmQZU0ylZQM1NO9aLl8qdN_wb5N4jeoOAqpO-gX0JMgugE-gjojfF3ZMMk34VMPeoVkqlHN4jeoGBCcyjq9N3X0BikCmTsvRDqY-xab57yqV893F55p-ZLVPKI7p9aXQvopE4wTpcL1jQbKFDd_-uGiopKiHhR0MItCmJHqd4hXqyZ_wb-CJcfVW7K2K8_CVdC6XhBKX38R7Hfo63MRz1wc7dg_6d3pWgbSHY-2VGyC0hCbJnfGV9HRinZzqEfMbH7XuU4tgsJBNsZnVgH8blz2WPTtWKPY3vNoWCtMHu8lxfrVjP5otRp8tSqPZY4LphoLGrrnBnYVMw-ptPM2vXOQSeqlQbHUeD1Ijg-408ce17gPIRRFIVhQF17EPxlWRo5dL3yVzRyQ391IfhPH9N1wnDtP3jRirrrgAauTzDklV2m38Mj7t_y5S-eWySY)

## Installation & Setup

### Prerequisites

- Python 3.12+
- API Keys for Google Gemini and/or OpenRouter.

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/DogRog/paper-check.git
   cd paper-check
   ```

2. **Install dependencies:**
   This project uses `pyproject.toml`. You can install dependencies using `pip`:

   ```bash
   pip install .
   ```

   Or if you are using `uv`:

   ```bash
   uv sync
   ```

3. **Environment Variables:**
   Create a `.env` file in the root directory and add your API keys:

   ```env
   GOOGLE_API_KEY=your_google_api_key
   OPENROUTER_API_KEY=your_openrouter_api_key
   ```

### Running the Application

Start the backend server (which also serves the frontend):

```bash
uvicorn backend.app:app --reload
```

The application will be available at `http://localhost:8000`.

## Model Finetuning (Optional)

The `model/` directory contains scripts to finetune a Qwen 3 14B model using the PeerRead dataset.

1. Navigate to the `model/` directory.

    ```bash
    cd model/
    ```

2. Download PeerRead dataset

    ```bash
    git clone https://github.com/allenai/PeerRead.git
    ```

3. Run the notebook to prepare the finetuning data:

   ```bash
   jupyter notebook prepare_finetune_data.ipynb
   ```

4. Run the finetuning script:

   ```bash
   bash run_full_finetuning.sh
   ```
5. Export the finetuned model for inference:

   ```bash
   bash export_model.sh
   ```

6. You can run the model at <https://endpoints.huggingface.co/> or run it locally if you have the necessary hardware.

## License

[Apache 2.0](LICENSE)
