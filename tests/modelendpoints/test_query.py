"""
Author: Jared Moore
Date: October, 2024

Tests for querying models.
"""

import asyncio
import json
import os
import signal
import unittest
from unittest.mock import patch, MagicMock

from modelendpoints.query import (
    wait_for_load,
    start_vllm_process,
    close_vllm_process,
    to_batch_json,
    process_batch_results,
    vllm_batch,
    openai_chat,
    anthropic_chat,
    get_model_source,
    get_option,
    find_answer,
    Endpoint,
)

TEST_MESSAGES = [{"role": "user", "content": "Hi how are you?"}]


class TestQueryModelsBasics(unittest.TestCase):

    @patch("requests.Session.get")
    def test_wait_for_load(self, mock_get):
        mock_get.return_value.json.return_value = {"status": "loaded"}
        wait_for_load("http://localhost:8000/v1/")
        mock_get.assert_called_once()

    # TODO: I have not tested these next two tests.
    @unittest.skipUnless(
        os.getenv("RUN_GPU_TESTS", "False") == "True", "Skipping gpu test case"
    )
    @patch("requests.Session.get")
    @patch("subprocess.Popen")
    @patch("torch.cuda.device_count", return_value=1)
    @patch("huggingface_hub.login")
    @patch.dict(
        os.environ,
        {"HF_HOME": "/tmp/test_hf_home"},
    )
    def test_start_vllm_process(
        self, mock_login, mock_device_count, mock_popen, mock_get
    ):
        mock_get.return_value.json.return_value = {"status": "loaded"}
        process = start_vllm_process(model="test-model")
        mock_popen.assert_called_once()
        self.assertIsInstance(process, MagicMock)

    @patch("subprocess.Popen")
    def test_close_vllm_process(self, mock_popen):
        # Create a mock instance of Popen
        process = mock_popen.return_value

        # Call the function with the mock process
        close_vllm_process(process)

        # Assert that the methods were called
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=60)
        process.kill.assert_called_once()

    def test_to_batch_json(self):
        msgs = [{"role": "user", "content": "Hello"}]
        keys_to_messages = {"key1": msgs}
        result = to_batch_json(keys_to_messages)
        self.assertIsInstance(result, bytes)
        request = json.loads(result.decode("utf-8"))
        self.assertEqual(request["body"]["messages"], msgs)
        self.assertEqual(request["custom_id"], "key1")

    def test_process_batch_results(self):
        response = {
            "custom_id": "key1",
            "response": {
                "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}]
            },
        }
        result = process_batch_results(json.dumps(response))
        self.assertIn("key1", result)
        self.assertEqual(result["key1"]["text"], "Hi")

        response = {
            "custom_id": "key1",
            "response": {
                "body": {
                    "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}]
                }
            },
        }
        result = process_batch_results(json.dumps(response))
        self.assertIn("key1", result)
        self.assertEqual(result["key1"]["text"], "Hi")

    def test_get_model_source(self):
        self.assertEqual(get_model_source("gpt-3"), "openai")
        self.assertEqual(get_model_source("claude-v1"), "anthropic")
        self.assertRaises(ValueError, get_model_source, "unknown-model")

    def test_get_option(self):
        self.assertEqual(get_option("(A)"), "A")
        self.assertEqual(get_option("B."), "B")
        self.assertEqual(get_option("▁▁▁▁▁▁A"), "A")

    def test_find_answer(self):
        answers = ["Answer A", "Answer B", "Answer C"]
        self.assertEqual(find_answer("A", answers), "Answer A")
        self.assertIsNone(find_answer("D", answers))

    @patch("modelendpoints.query.start_vllm_process")
    @patch("modelendpoints.query.close_vllm_process")
    @patch("tempfile.NamedTemporaryFile")
    def test_vllm_batch(self, mock_tempfile, mock_close, mock_start):
        keys_to_messages = {"key1": [{"role": "user", "content": "Hello"}]}
        response = {
            "custom_id": "key1",
            "response": {
                "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}]
            },
        }

        # Mock the temporary file
        mock_temp_file = MagicMock()
        mock_temp_file.__enter__.return_value.name = "mocked_temp_file"
        mock_temp_file.__enter__.return_value.read.return_value = json.dumps(
            response
        ).encode("utf-8")
        mock_tempfile.return_value = mock_temp_file

        result = vllm_batch(keys_to_messages, model="test-model")

        self.assertIn("key1", result)
        self.assertEqual(result["key1"]["text"], "Hi")

    # Mocking OpenAI client
    @patch("openai.OpenAI")
    def test_openai_chat(self, mock_openai):
        client = mock_openai.return_value
        client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="Hi"))
        ]
        messages = [{"role": "user", "content": "Hello"}]
        result = openai_chat(client, messages)
        self.assertEqual(result["text"], "Hi")

    @patch("together.Together")
    def test_together_chat(self, mock_openai):
        client = mock_openai.return_value
        client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="Hi"))
        ]
        messages = [{"role": "user", "content": "Hello"}]
        result = openai_chat(client, messages)
        self.assertEqual(result["text"], "Hi")

    # Mocking Anthropic client
    @patch("anthropic.Anthropic")
    def test_anthropic_chat(self, mock_anthropic):
        client = mock_anthropic.return_value
        client.messages.create.return_value.content = [MagicMock(text="Hi")]
        messages = [{"role": "user", "content": "Hello"}]
        result = anthropic_chat(client, messages)
        self.assertEqual(result["text"], "Hi")


class TestBatchProcessing(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.test_messages = [{"role": "user", "content": "Test message"}]
        self.keys_to_messages = {
            "1": self.test_messages,
            "2": self.test_messages,
            "3": self.test_messages,
        }

    @patch("openai.OpenAI")
    def test_batch_serial_processing(self, mock_openai):
        # Mock successful API responses
        client = mock_openai.return_value
        client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="Response"))
        ]

        with Endpoint(
            source="openai",
            model="gpt-3.5-turbo",
            batch_prompts=True,
            serial_batch_prompts=True,
        ) as endpoint:
            response = endpoint(
                keys_to_messages=self.keys_to_messages, temperature=0, max_tokens=10
            )

            self.assertEqual(len(response), 3)
            self.assertEqual(response["1"]["text"], "Response")
            self.assertEqual(response["2"]["text"], "Response")
            self.assertEqual(response["3"]["text"], "Response")

    @patch("openai.OpenAI")
    def test_batch_serial_with_errors(self, mock_openai):
        # Mock API responses with some failures
        client = mock_openai.return_value
        calls = 0
        import openai

        def side_effect(*_, **__):
            nonlocal calls
            calls += 1
            if calls == 2:  # Fail on second call
                error = openai.APIError(
                    message="API Error",
                    request=MagicMock(
                        method="post", url="https://api.openai.com/v1/chat/completions"
                    ),
                    body={},
                )
                raise error
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
            return mock_response

        client.chat.completions.create.side_effect = side_effect

        with Endpoint(
            source="openai",
            model="gpt-3.5-turbo",
            batch_prompts=True,
            serial_batch_prompts=True,
        ) as endpoint:
            response = endpoint(
                keys_to_messages=self.keys_to_messages, temperature=0, max_tokens=10
            )

            self.assertEqual(len(response), 2)  # Only successful responses
            self.assertEqual(response["1"]["text"], "Response")
            self.assertEqual(response["3"]["text"], "Response")
            self.assertNotIn("2", response)

    @patch("openai.AsyncOpenAI")
    async def test_batch_async_with_keyboard_interrupt(self, mock_openai):
        # Mock API responses with KeyboardInterrupt
        client = mock_openai.return_value
        calls = 0

        async def mock_create(*_, **__):
            nonlocal calls
            calls += 1
            if calls == 2:
                await asyncio.sleep(0.1)
                # Simulate signal by setting shutdown event
                # This is better than raising KeyboardInterrupt directly
                os.kill(os.getpid(), signal.SIGINT)
                # Allow some time for signal handler
            await asyncio.sleep(0.2)
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
            return mock_response

        client.chat.completions.create.side_effect = mock_create

        with Endpoint(
            source="openai",
            model="gpt-3.5-turbo",
            batch_prompts=True,
            async_function=True,
            serial_batch_prompts=False,
        ) as endpoint:
            response = await endpoint(
                keys_to_messages=self.keys_to_messages, temperature=0, max_tokens=10
            )

            # Should have partial results
            self.assertLess(len(response), 3)
            self.assertIn("1", response)
            self.assertEqual(response["1"]["text"], "Response")

    @patch("openai.AsyncOpenAI")
    async def test_batch_async_cleanup_after_shutdown(self, mock_openai):
        # Test that resources are properly cleaned up after shutdown
        client = mock_openai.return_value

        async def mock_create(*_, **__):
            os.kill(os.getpid(), signal.SIGINT)
            await asyncio.sleep(0.1)
            return MagicMock(choices=[MagicMock(message=MagicMock(content="Response"))])

        client.chat.completions.create.side_effect = mock_create

        with Endpoint(
            source="openai",
            model="gpt-3.5-turbo",
            batch_prompts=True,
            async_function=True,
            serial_batch_prompts=False,
        ) as endpoint:
            # Get initial task count
            initial_tasks = {
                t
                for t in asyncio.all_tasks()
                if not t.done() and t != asyncio.current_task()
            }

            _ = await endpoint(
                keys_to_messages=self.keys_to_messages, temperature=0, max_tokens=10
            )

            # Get final task count and filter out the test task
            final_tasks = {
                t
                for t in asyncio.all_tasks()
                if not t.done() and t != asyncio.current_task()
            }

            # Check that no new tasks were added
            new_tasks = final_tasks - initial_tasks
            self.assertEqual(
                len(new_tasks),
                0,
                f"Found {len(new_tasks)} uncompleted tasks after shutdown",
            )


class TestEndpoints(unittest.IsolatedAsyncioTestCase):

    @patch("openai.OpenAI")
    def test_endpoint_openai(self, mock_openai):
        with Endpoint(source="openai", model="gpt-3") as endpoint:
            self.assertIsNotNone(endpoint)

    @patch("together.Together")
    def test_endpoint_together(self, mock_openai):
        with Endpoint(source="together", model="meta-llama/Llama-3.1-70B") as endpoint:
            self.assertIsNotNone(endpoint)

    @patch("anthropic.Anthropic")
    def test_endpoint_anthropic(self, mock_anthropic):
        with Endpoint(source="anthropic", model="claude-v1") as endpoint:
            self.assertIsNotNone(endpoint)

    @patch("modelendpoints.query.close_vllm_process")
    @patch("openai.OpenAI")
    @patch("modelendpoints.query.start_vllm_process")
    def test_endpoint_vllm(self, mock_start, mock_openai, mock_close):
        with Endpoint(source="vllm", model="meta-llama/Llama-3.1-70B") as endpoint:
            self.assertIsNotNone(endpoint)

    @unittest.skipUnless(
        os.getenv("RUN_QUERY_TESTS", "False") == "True", "Skipping query test case"
    )
    def test_actual_openai(self):
        with Endpoint(
            source="openai", model="gpt-3.5-turbo", temperature=0, max_tokens=10
        ) as endpoint:
            response = endpoint(messages=TEST_MESSAGES)
            response_text = response.get("text")
            self.assertIsNotNone(response_text)

    @unittest.skipUnless(
        os.getenv("RUN_QUERY_TESTS", "False") == "True", "Skipping query test case"
    )
    def test_actual_openai_batch_loop(self):
        with Endpoint(
            source="openai",
            model="gpt-3.5-turbo",
            batch_prompts=True,
        ) as endpoint:

            keys_to_messages = {"1": TEST_MESSAGES}
            response = endpoint(
                keys_to_messages=keys_to_messages, temperature=0, max_tokens=10
            )
            self.assertIsNotNone(response["1"]["text"])

    @unittest.skipUnless(
        os.getenv("RUN_BATCH_TESTS", "False") == "True", "Skipping batch test case"
    )
    def test_actual_openai_batch(self):
        with Endpoint(
            source="openai",
            model="gpt-3.5-turbo",
            batch_prompts=True,
            batch_function=True,
        ) as endpoint:

            keys_to_messages = {"1": TEST_MESSAGES}
            response = endpoint(
                keys_to_messages=keys_to_messages, temperature=0, max_tokens=10
            )
            self.assertIsNotNone(response["1"]["text"])

    @unittest.skipUnless(
        os.getenv("RUN_QUERY_TESTS", "False") == "True", "Skipping query test case"
    )
    async def test_actual_openai_batch_loop_async(self):
        with Endpoint(
            source="openai",
            model="gpt-3.5-turbo",
            batch_prompts=True,
            async_function=True,
        ) as endpoint:

            keys_to_messages = {"1": TEST_MESSAGES}
            response = await endpoint(
                keys_to_messages=keys_to_messages, temperature=0, max_tokens=10
            )
            self.assertIsNotNone(response["1"]["text"])

    @unittest.skipUnless(
        os.getenv("RUN_QUERY_TESTS", "False") == "True", "Skipping query test case"
    )
    def test_actual_anthropic(self):
        with Endpoint(
            source="anthropic",
            model="claude-3-haiku-20240307",
            temperature=0,
            max_tokens=10,
        ) as endpoint:
            response = endpoint(messages=TEST_MESSAGES)
            response_text = response.get("text")
            self.assertIsNotNone(response_text)

    @unittest.skipUnless(
        os.getenv("RUN_QUERY_TESTS", "False") == "True", "Skipping query test case"
    )
    async def test_actual_anthropic_batch(self):
        with Endpoint(
            source="anthropic",
            model="claude-3-haiku-20240307",
            batch_prompts=True,
            async_function=True,
        ) as endpoint:

            keys_to_messages = {"1": TEST_MESSAGES}
            response = await endpoint(
                keys_to_messages=keys_to_messages, temperature=0, max_tokens=10
            )
            self.assertIsNotNone(response["1"]["text"])

        with self.assertRaises(ValueError):
            with Endpoint(
                model="claude-3-haiku-20240307",
                batch_prompts=True,
            ) as endpoint:
                keys_to_messages = {"1": []}
                kwargs = {"max_tokens": 256, "system": "Test"}
                response = await endpoint(keys_to_messages=keys_to_messages, **kwargs)

        with self.assertRaises(ValueError):
            with Endpoint(
                model="claude-3-haiku-20240307",
                batch_prompts=True,
                async_function=True,
            ) as endpoint:
                keys_to_messages = {"1": []}
                kwargs = {"max_tokens": 256, "system": "Test"}
                response = await endpoint(keys_to_messages=keys_to_messages, **kwargs)

    @unittest.skipUnless(
        os.getenv("RUN_QUERY_TESTS", "False") == "True", "Skipping query test case"
    )
    async def test_actual_together_batch(self):
        with Endpoint(
            source="together",
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            batch_prompts=True,
            async_function=True,
        ) as endpoint:

            keys_to_messages = {"1": TEST_MESSAGES}
            response = await endpoint(
                keys_to_messages=keys_to_messages, temperature=0, max_tokens=10
            )
            self.assertIsNotNone(response["1"]["text"])

    @unittest.skipUnless(
        os.getenv("RUN_QUERY_TESTS", "False") == "True", "Skipping query test case"
    )
    def test_actual_together(self):
        with Endpoint(
            source="together",
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            temperature=0,
            max_tokens=10,
        ) as endpoint:
            response = endpoint(messages=TEST_MESSAGES)
            response_text = response.get("text")
            self.assertIsNotNone(response_text)

    # TODO: I have not tested these next two tests.
    @unittest.skipUnless(
        os.getenv("RUN_GPU_TESTS", "False") == "True", "Skipping gpu test case"
    )
    def test_real_vllm_process(self):
        with Endpoint(
            source="vllm",
            model="meta-llama/Llama-3.1-8B",
            temperature=0,
            max_tokens=10,
        ) as endpoint:
            response = endpoint(messages=TEST_MESSAGES)
            response_text = response.get("text")
            self.assertIsNotNone(response_text)

    @unittest.skipUnless(
        os.getenv("RUN_GPU_TESTS", "False") == "True", "Skipping gpu test case"
    )
    def test_actual_vllm_batch(self):
        with Endpoint(
            source="vllm",
            model="meta-llama/Llama-3.1-8B",
            batch_prompts=True,
            batch_function=True,
        ) as endpoint:
            keys_to_messages = {"1": TEST_MESSAGES}
            response = endpoint(
                keys_to_messages=keys_to_messages, temperature=0, max_tokens=10
            )
            self.assertIsNotNone(response["1"]["text"])

    @unittest.skipUnless(
        os.getenv("RUN_GPU_TESTS", "False") == "True", "Skipping gpu test case"
    )
    def test_actual_vllm_batch_loop(self):
        with Endpoint(
            source="vllm",
            model="meta-llama/Llama-3.1-8B",
            batch_prompts=True,
        ) as endpoint:
            keys_to_messages = {"1": TEST_MESSAGES}
            response = endpoint(
                keys_to_messages=keys_to_messages, temperature=0, max_tokens=10
            )
            self.assertIsNotNone(response["1"]["text"])


if __name__ == "__main__":
    unittest.main()
