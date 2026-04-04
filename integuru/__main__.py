from dotenv import load_dotenv
import time  # Add this import

load_dotenv()

from integuru.main import call_agent
import asyncio
import click

@click.command()
@click.option(
    "--model", default="gpt-4o", help="The LLM model to use (default is gpt-4o)"
)
@click.option("--prompt", required=True, help="The prompt for the model")
@click.option(
    "--har-path",
    default="./network_requests.har",
    help="The HAR file path (default is ./network_requests.har)",
)
@click.option(
    "--cookie-path",
    default="./cookies.json",
    help="The cookie file path (default is ./cookies.json)",
)
@click.option(
    "--max_steps", default=20, type=int, help="The max_steps (default is 20)"
)
@click.option(
    "--input_variables",
    multiple=True,
    type=(str, str),
    help="Input variables in the format key value",
)
@click.option(
    "--generate-code",
    is_flag=True,
    default=False,
    help="Whether to generate the full integration code",
)
def cli(
    model, prompt, har_path, cookie_path, max_steps, input_variables, generate_code
):
    input_vars = dict(input_variables)
    asyncio.run(
        call_agent(
            model,
            prompt,
            har_path,
            cookie_path,
            input_variables=input_vars,
            max_steps=max_steps,
            to_generate_code=generate_code,
        )
    )

if __name__ == "__main__":
    cli()
