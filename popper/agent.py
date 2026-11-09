# Standard Library Imports
import contextlib
import io
import logging
import os
import re
import sys
import json
import traceback
import multiprocessing

# Third-Party Imports
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2

# Typing and Pydantic
from typing import (
    Optional, List, Tuple, Union, Literal, Dict, TypedDict, Annotated
)
from pydantic import BaseModel, Field

# LangChain and LangGraph Imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.interactive_env import is_interactive_env
from langchain_core.messages.base import get_msg_title_repr
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Logging Configuration
logging.getLogger("httpx").setLevel(logging.WARNING)

from .utils import get_llm, pretty_print
from .prompt_utils import *
from popper.react_agent import ReactAgent

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Execution surpassed time limit.")

def parse_output(solution):
    """When we add 'include_raw=True' to structured output,
    it will return a dict w 'raw', 'parsed', 'parsing_error'."""
    if not solution["parsed"]:
        print('code solution fail to produce')
        print(solution)
    return solution["parsed"]

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error : Binary flag for control flow to indicate whether test error was tripped
        messages : With user question, error messages, reasoning
        generation : Code solution
        iterations : Number of tries
    """

    error: str
    messages: List
    generation: str
    iterations: int
    status: str
    captured_output: str
    p_val: float

class code(BaseModel):
    """Code output"""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")

class test_specification(BaseModel):
    """tool_specification."""

    test_name: Optional[str] = Field(description="name of the test")
    test_description: Optional[str] = Field(description="test description")
    null_hypothesis: Optional[str] = Field(description="null hypothesis")
    alternate_hypothesis: Optional[str] = Field(description="alternate hypothesis")

class LogLikelihoodRatioInput(BaseModel):
    likelihood_h1: float = Field(description="probability of data given hypothesis is alternative, P(data|h1)")
    likelihood_h0: float = Field(description="probability of data given hypothesis is null, P(data|h0)")

class parser_yes_no(BaseModel):
    """does the given text have p-value? if so, what is the p-value?"""

    check_output_error: Optional[str] = Field(
        description="Does the given text contains a p-value? Yes if it has; No if does not. "
    )
    p_val: Optional[str] = Field(description="p-value formatted in scientific notations")

class data_input_check_result(BaseModel):
    """does the code make up fake data entries"""

    fake_data_entries: str = Field(
        description="Does the code make up fake data entries? Yes if it does; No if does not. "
    )

class relevance_subhypothesis(BaseModel):
    """Is the subhypothesis relevant to the main hypothesis?"""

    relevance_reasoning: Optional[str] = Field(
        description="What is the reason behind this relevance score?"
    )
    relevance_score: Optional[str] = Field(description="relevance score")


class OutputSpecification(BaseModel):
    """Output specification for the hypothesis testing."""

    main_hypothesis: Optional[str] = Field(description="The main hypothesis under study")
    falsification_test_result: Optional[str] = Field(description="The result of the sequential falsification test")
    reasoning: Optional[str] = Field(description="Reasoning, summarizing, and analyzing these results")
    conclusion: Optional[bool] = Field(description="Conclusion on whether the hypothesis is true or false (True/False)")
    rationale: Optional[str] = Field(description="Rationale behind the conclusion")

def likelihood_ratio_e_value(likelihood_ratio, alpha=0.1):
    likelihood_ratio = np.array(likelihood_ratio)
    cum_e = 1/np.prod(likelihood_ratio)
    if cum_e < alpha:
        return True, cum_e
    else:
        return False, cum_e

def e_value_kappa_calibrator(p_values, alpha=0.1, kappa = 0.5):
    p_values = np.array(p_values)
    e_values = kappa * p_values ** (kappa-1)
    cum_e = np.prod(e_values)

    if cum_e > 1/alpha:
        return True, cum_e
    else:
        return False, cum_e

def e_value_integral_calibrator(p_values, alpha=0.1):
    p_values = np.array(p_values)
    e_values = (1 - p_values + p_values * np.log(p_values))/(p_values * (-np.log(p_values))**2)
    cum_e = np.prod(e_values)

    if cum_e > 1/alpha:
        return True, cum_e
    else:
        return False, cum_e

def fishers_method(p_values, alpha=0.1):
    p_values = np.array(p_values)
    # Apply Fisher's method formula
    chi_square_stat = -2 * np.sum(np.log(p_values))
    # Degrees of freedom is twice the number of p-values
    degrees_of_freedom = 2 * len(p_values)
    # Calculate the combined p-value from the chi-square distribution
    combined_p_value = 1 - chi2.cdf(chi_square_stat, degrees_of_freedom)
    
    if combined_p_value < alpha:
        return True, combined_p_value
    else:
        return False, combined_p_value
    
def p_val_to_log_likelihood_ratio(p_val):
    """
    Given the p-value, 
    return the log likelihood ratio.
    """
    return -np.log(p_val)

# def sequential_probability_ratio_test(list_of_log_likelihood_ratio: List[float], alpha=0.1, beta=0.1):
#     """
#     Given a list of log likelihood ratios, conduct the sequential probability ratio test at alpha = 0.1, and beta= 0.1.
#     Return whether to pass, continue, or fail the test.
#     """
#     cumulative_lr = sum(list_of_log_likelihood_ratio)

#     A = np.log(beta/(1-alpha))
#     B = np.log((1-beta)/alpha)

#     if cumulative_lr > B:
#         return "sufficient evidence - PASS", cumulative_lr
#     elif cumulative_lr < A:
#         return "sufficient evidence for null hypothesis - FAIL", cumulative_lr
#     else:ƒs
#         return "insufficient evidence - CONTINUE", cumulative_lr


class falsification_test_coding_agent:

    def __init__(self, data, llm = "claude-3-5-sonnet-20241022", max_retry = 10, time_limit = 10, reflect = True, verbose = True, llm_approx = False, domain="biology", port=None, api_key="EMPTY", base_url=None):
        self.data = data
        self.llm = get_llm(llm, temperature=0.0, port=port, api_key=api_key, base_url=base_url)
        print(llm)
        self.time_limit = time_limit
        self.llm_approx = llm_approx
        self.domain = domain

        self.format_check_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system", "You are an evaluateGPT. Check if the output from a statistical test contains a p-value. If it does not have a p-value, then return No. If it return p value is nan, also return No. Otherwise Yes. Test output: ",
                ),
                ("placeholder", "{messages}"),
            ]
        )
    
        
        self.tool_404_parser_llm = self.format_check_prompt | self.llm.with_structured_output(parser_yes_no)
        
        self.data_check_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system", "You are an evaluateGPT. Your task is to check if a LLM-generated code is hallucinating fake data entries. If code is directly using an existing datafrmae, then return No. However, if the code is making up new data entries such as `df = pd.DataFrame({{fake_data_entries}})`, return Yes.",
                ),
                ("placeholder", "{messages}"),
            ]
        )
        self.data_checker = self.data_check_prompt | self.llm.with_structured_output(data_input_check_result)
        
        self.max_retry = max_retry
        if reflect:
            self.reflect = "reflect"
        else:
            self.reflect = "do not reflect"
        
        self.verbose = verbose

        system_prompt = get_coding_agent_system_prompt(self.llm_approx, self.domain)
        print('system_prompt: ', system_prompt)
        code_gen_prompt_claude = ChatPromptTemplate.from_messages(
            [
                (
                    "system", system_prompt,
                ),
                ("placeholder", "{messages}"),
            ]
        )

        structured_llm_claude = self.llm.with_structured_output(code, include_raw=True)

        code_gen_chain = code_gen_prompt_claude | structured_llm_claude | parse_output

        max_iterations = self.max_retry
        flag = self.reflect


        def generate(state: GraphState):
            """
            Generate a code solution

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): New key added to state, generation
            """
            if self.verbose:
                print("---GENERATING CODE SOLUTION---")

            # State
            messages = state["messages"]
            iterations = state["iterations"]
            error = state["error"] if "error" in state else None

            # We have been routed back to generation with an error
            if error == "yes":
                messages += [
                    (
                        "user",
                        "Now, try again. Invoke the code tool to structure the output with a prefix, imports and code block.",
                    )
                ]

            # Solution

            for try_ in range(20):
                code_solution = code_gen_chain.invoke(
                    {"context": self.data, "messages": messages}
                )
                if code_solution:
                    break
                else:
                    print('NoneType code_solution')

            messages += [
                (
                    "assistant",
                    f"{code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}",
                )
            ]

            # Increment
            iterations = iterations + 1
            return {"generation": code_solution, "messages": messages, "iterations": iterations}


        def code_check(state: GraphState):
            """
            Check code

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): New key added to state, error
            """
            if self.verbose:
                print("---CHECKING CODE---")

            # State
            messages = state["messages"]
            code_solution = state["generation"]
            iterations = state["iterations"]

            # Get solution components
            imports = code_solution.imports
            code = code_solution.code

            print(imports + '\n\n' + code)

            # Check imports
            try:
                exec(imports)
            except Exception as e:
                if self.verbose:
                    print("---CODE IMPORT CHECK: FAILED---")
                error_message = [("user", f"Your solution failed the import test: {e}")]
                messages += error_message
                return {
                    "generation": code_solution,
                    "messages": messages,
                    "iterations": iterations,
                    "error": "yes",
                    "status": "Failed test"
                }
            
            data_check = self.data_checker.invoke({ "messages": [("user", imports + '\n\n' + code)]}).dict()
            if data_check['fake_data_entries'].lower() == "yes":
                print("Data input check failed")
                messages += [("user", "Your solution failed the data input test: Do NOT make up fake data entries.")]
                return {
                    "generation": code_solution,
                    "messages": messages,
                    "iterations": iterations,
                    "error": "yes",
                    "status": "Failed test"
                }

            # Set up logging and output capture as before
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
            logging.getLogger().setLevel(logging.INFO)
            
            # Function to run the code and capture the exception and output within the thread
            def run_code(queue):
                output_capture = io.StringIO()
                try:
                    full_code = imports + '\n\n' + code
                    exec_globals = globals().copy()
                    exec_globals.update(__builtins__)

                    with contextlib.redirect_stdout(output_capture), contextlib.redirect_stderr(output_capture):
                        logging.getLogger().handlers[0].stream = output_capture
                        exec(full_code, exec_globals)
                except Exception as e:
                    # Capture the exception, format the traceback, and put it in the queue
                    error_message = f"{traceback.format_exc()}"
                    queue.put(error_message)  # Place error message in queue for retrieval
                else:
                    # Put successful output in the queue if no error occurred
                    queue.put(output_capture.getvalue())

            queue = multiprocessing.Queue()
            process = multiprocessing.Process(target=run_code, args=(queue,))
            process.start()
            process.join(timeout=self.time_limit * 60)

            if process.is_alive():
                print("Process is taking too long... terminating.")
                process.terminate()
                process.join()  # Ensure process is cleaned up properly after termination
                if self.verbose:
                    print("---CODE BLOCK CHECK: FAILED---")
                    print("Execution surpassed time limit, please come up with a more efficient implementation.")
                    error_message = [("user", "Your solution failed due to time limit: Execution surpassed time limit, please come up with a more efficient implementation.")]
                    messages += error_message
                    return {
                        "generation": code_solution,
                        "messages": messages,
                        "iterations": iterations,
                        "error": "yes",
                        "status": "Failed test"
                    }
            else:
                # Check if queue contains an error message or the actual output
                if not queue.empty():
                    captured_output = queue.get()
                    if "Traceback" in captured_output:  # Check if result is an error
                        print("---CODE BLOCK CHECK: FAILED---")
                        print(captured_output)  # Print the error message
                        messages += [("user", f"Your solution failed the code execution test: {captured_output}")]
                        return {
                            "generation": code_solution,
                            "messages": messages,
                            "iterations": iterations,
                            "error": "yes",
                            "status": "Failed test"
                        }
                    else:
                        print("Process completed within the time limit.")
                        print("Captured Output:", captured_output)
                else:
                    print("No output was captured.")
                    messages += [("user", "Your solution failed the code execution test: No output was captured.")]
                    return {
                        "generation": code_solution,
                        "messages": messages,
                        "iterations": iterations,
                        "error": "yes",
                        "status": "Failed test"
                    }

            if self.llm_approx:
                #print("Captured output: ", captured_output)
                if len(captured_output) == 0:
                    if self.verbose:
                        print("---NO OUTPUT RETURNED---")
                    error_message = [("user", f"Your solution failed the output test, which tests if the result returns any thing for falsification test: " + captured_output)]
                    messages += error_message
                    return {
                        "generation": code_solution,
                        "messages": messages,
                        "iterations": iterations,
                        "error": "yes",
                        "status": "Failed test"
                    }
                else:
                    if self.verbose:
                        print("---NO CODE TEST FAILURES---")

                    return {
                        "generation": code_solution,
                        "messages": messages,
                        "iterations": iterations,
                        "error": "no",
                        "status": "success",
                        "captured_output": captured_output
                    }

            else:
                #print("Captured output: ", captured_output)
                # check if example produces reasonable output, or just return no result found
                if len(captured_output) == 0:
                    output = 'No'
                else:      
                    checker = self.tool_404_parser_llm.invoke({ "messages": [("user", captured_output)]}).dict()
                    output = checker['check_output_error']
                    try:
                        p_val = float(checker['p_val'])
                    except:
                        print('Error parsing p_value')
                        print(checker['p_val'])
                        output = 'No'
                    
                if output == 'No':
                    if self.verbose:
                        print("---P-value OUTPUT CHECK: FAILED---")
                    error_message = [("user", f"Your solution failed the output test, which tests if the result returns any p-value for falsification test: " + captured_output)]
                    messages += error_message
                    return {
                        "generation": code_solution,
                        "messages": messages,
                        "iterations": iterations,
                        "error": "yes",
                        "status": "Failed test"
                    }

                p_val = float(checker['p_val'])

                if np.isnan(p_val):
                    if self.verbose:
                        print("---P-value is nan: FAILED---")
                    error_message = [("user", f"Your solution p-value for falsification test is nan: " + captured_output)]
                    messages += error_message
                    return {
                        "generation": code_solution,
                        "messages": messages,
                        "iterations": iterations,
                        "error": "yes",
                        "status": "Failed test"
                    }

                if p_val == 0:
                    if self.verbose:
                        print("---P-value is 0: FAILED---")
                    error_message = [("user", f"Your solution p-value for falsification test is exact 0 - supposedly wrong: " + captured_output)]
                    messages += error_message
                    return {
                        "generation": code_solution,
                        "messages": messages,
                        "iterations": iterations,
                        "error": "yes",
                        "status": "Failed test"
                    }

                # No errors
                if self.verbose:
                    print("---NO CODE TEST FAILURES---")
                return {
                    "generation": code_solution,
                    "messages": messages,
                    "iterations": iterations,
                    "error": "no",
                    "status": "success",
                    "captured_output": captured_output,
                    "p_val": checker['p_val']
                }


        def reflect(state: GraphState):
            """
            Reflect on errors

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): New key added to state, generation
            """
            if self.verbose:
                print("---GENERATING CODE SOLUTION---")

            # State
            messages = state["messages"]
            iterations = state["iterations"]
            code_solution = state["generation"]

            # Prompt reflection

            # Add reflection
            reflections = code_gen_chain.invoke(
                {"context": self.data, "messages": messages}
            )
            messages += [("assistant", f"Here are reflections on the error: {reflections}")]
            return {"generation": code_solution, "messages": messages, "iterations": iterations}


        ### Edges


        def decide_to_finish(state: GraphState):
            """
            Determines whether to finish.

            Args:
                state (dict): The current graph state

            Returns:
                str: Next node to call
            """
            error = state["error"]
            iterations = state["iterations"]

            if error == "no" or iterations == max_iterations:
                if self.verbose:
                    print("---DECISION: FINISH---")
                return "end"
            else:
                if self.verbose:
                    print("---DECISION: RE-TRY SOLUTION---")
                if flag == "reflect":
                    return "reflect"
                else:
                    return "generate"

        from langgraph.graph import END, StateGraph

        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("generate", generate)  # generation solution
        workflow.add_node("check_code", code_check)  # check code
        workflow.add_node("reflect", reflect)  # reflect

        # Build graph
        workflow.set_entry_point("generate")
        workflow.add_edge("generate", "check_code")
        workflow.add_conditional_edges(
            "check_code",
            decide_to_finish,
            {
                "end": END,
                "reflect": "reflect",
                "generate": "generate",
            },
        )
        workflow.add_edge("reflect", "generate")
        self.app = workflow.compile()

    def go(self, question, log = None):
        print(question)
        self.question = question
        config = {"recursion_limit": 500}
        graph = self.app.invoke({"messages": [("user", question)], "iterations": 0}, config = config)
        #solution = graph["generation"]
        return graph

class falsification_test_react_agent:
    def __init__(self, data_loader, llm = "claude-3-5-sonnet-20241022", max_retry = 10, domain="biology", prompt_revision = False, port=None, api_key="EMPTY", base_url=None):
        self.data_loader = data_loader
        self.llm = get_llm(llm, temperature=0.0, port=port, api_key=api_key, base_url=base_url)
        self.domain = domain
        self.max_retry = max_retry
        
        self.agent = ReactAgent(
            model_name=llm,
            prompt_revision=prompt_revision,
            port=port,
            api_key=api_key,
            base_url=base_url,
        )
        
        self.pvalue_check_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system", "You are an evaluateGPT. Check if the output from a statistical test contains a p-value. If it does not have a p-value, then return No. If it return p value is nan, also return No. Otherwise Yes. Test output: ",
                ),
                ("placeholder", "{messages}"),
            ]
        )
    
        
        self.pvalue_parser = self.pvalue_check_prompt | self.llm.with_structured_output(parser_yes_no)
    
    def go(self, question, log = None):
        def parse_falsification_test(input_string):
            # Define the regex pattern to capture each field, allowing for variable spacing
            pattern = (
                r"Falsification Test name:\s*(.*?)\s*"
                r"Falsification Test description:\s*(.*?)\s*"
                r"Falsification Test Null sub-hypothesis:\s*(.*?)\s*"
                r"Falsification Test Alternate sub-hypothesis:\s*(.*)"
            )
            
            match = re.search(pattern, input_string, re.DOTALL)
            if not match:
                return None
            
            test_name = match.group(1).strip()
            test_description = match.group(2).strip()
            null_hypothesis = match.group(3).strip()
            alternate_hypothesis = match.group(4).strip()
            
            falsification_test_json = {
                "Falsification Test name": test_name,
                "Falsification Test description": test_description,
                "Falsification Test Null hypothesis": null_hypothesis,
                "Falsification Test Alternate hypothesis": alternate_hypothesis
            }
            
            return falsification_test_json
        
        log = log if log is not None else {'executor': []}
        self.last_attempt_audit = []

        def failed(reason):
            return {"error": "yes", "status": "Failed test", "captured_output": None,
                    "p_val": None, "failure_reason": reason, "attempts": self.last_attempt_audit}

        falsification_test = parse_falsification_test(question)
        
        if not falsification_test:
            print("Failed to parse Falsification Test")
            print(question)
            return failed("invalid_test_proposal")
        
        test_spec = json.dumps(falsification_test, indent=4)
        
        for attempt_index in range(self.max_retry):
            attempt = {"attempt": attempt_index + 1, "status": "started"}
            self.last_attempt_audit.append(attempt)
            tool = self.agent.agent.tools[0]
            execution_start = len(tool.executions)
            reply_start = len(self.agent.protocol_audit)
            try:
                captured_output = self.agent.generate(self.data_loader, test_spec, self.domain, log)
                if not captured_output:
                    attempt.update(status="failed", reason="execution_error",
                                   error=getattr(self.agent, 'last_error', None))
                    print("---No Captured Output---")
                    print("---DECISION: RE-TRY SOLUTION---")
                    log['executor'].append("No Captured Output - Retry Solution")
                    continue
                
                if self.agent.execution_mode != 'legacy':
                    # Guarded executors return host-registered evidence. Never
                    # re-extract or replace this number from assistant prose.
                    if self.agent.last_result is None:
                        raise ValueError('No execution-backed test result')
                    parsed_output = {'check_output_error': 'Yes',
                                     'p_val': str(self.agent.last_result['p_value'])}
                else:
                    try:
                        # Structured-output repair operates on this same output.
                        # Its exhaustion must not draw another whole analysis.
                        parsed_output = self.pvalue_parser.invoke({ "messages": [("user", captured_output)]}).dict()
                    except Exception as exc:
                        attempt.update(status="failed", reason="pvalue_parse_error",
                                       error={"type": type(exc).__name__, "message": str(exc)})
                        return failed("pvalue_parse_error")
                
                if 'check_output_error' in parsed_output and parsed_output['check_output_error'] and parsed_output['check_output_error'].strip().lower() == "no":
                    print("---P-value OUTPUT CHECK: FAILED---")
                    log['executor'].append("P-value OUTPUT CHECK: FAILED - Analysis will not be rerun")
                    attempt.update(status="failed", reason="no_valid_pvalue")
                    return failed("no_valid_pvalue")
                try:
                    p_val = float(parsed_output['p_val'])
                    if not np.isfinite(p_val) or not 0 < p_val <= 1:
                        raise ValueError("p-value must be finite and in (0, 1]")
                except (ValueError, TypeError, KeyError) as exc:
                    attempt.update(status="failed", reason="invalid_pvalue", error=str(exc))
                    return failed("invalid_pvalue")

                attempt.update(status="success", p_value=p_val)
                if self.agent.execution_mode == 'legacy':
                    from popper.execution_audit import audit_legacy_pvalue
                    attempt['legacy_pvalue_audit'] = audit_legacy_pvalue(
                        p_val, tool.executions[execution_start:], self.agent.protocol_audit[reply_start:])
                else:
                    attempt['registered_result'] = dict(self.agent.last_result)

                # No errors
                print("---NO CODE TEST FAILURES---")
                print("---DECISION: FINISH---")
                log['executor'].append("NO CODE TEST FAILURES - FINISH")
                log['executor'].append(f"P-value: {p_val}")
                return {
                    "error": "no",
                    "status": "success",
                    "captured_output": captured_output,
                    "p_val": parsed_output['p_val'],
                    "attempts": self.last_attempt_audit,
                    "fallback_reason": (self.agent.last_result or {}).get('fallback_reason'),
                }
                    
            except Exception as e:
                attempt.update(status="failed", reason="execution_error",
                               error={"type": type(e).__name__, "message": str(e)})
                print(f"Falsification Test failed with the following error: {e}")
                print(traceback.format_exc())
                print("---DECISION: RE-TRY SOLUTION---")
                log['executor'].append(f"Falsification Test failed with the following error: {e}")
        
        print(f"Falsification Test exceeded maximum number of retries; max_retries={self.max_retry}")
        print("---DECISION: FINISH---")
        log['executor'].append(f"Falsification Test exceeded maximum number of retries; max_retries={self.max_retry}")
        return failed("execution_retries_exhausted")
        

class likelihood_estimation_agent:
    def __init__(self, llm = 'claude-3-5-sonnet-20241022', port=None, api_key="EMPTY", base_url=None):
        self.llm = get_llm(llm, port=port, api_key=api_key, base_url=base_url)
        self.output_parser = self.llm.with_structured_output(LogLikelihoodRatioInput)

    def go(self, main_hypothesis, falsification_test, data):
        prompt_modifier = get_likelihood_estimation_agent_prompt(main_hypothesis, falsification_test, data)

        #print(prompt_modifier)
        self.app = create_react_agent(self.llm, [])

        config = {"recursion_limit": 500}
        inputs = {"messages": [("user", prompt_modifier)]}
        log = []
        for s in self.app.stream(inputs, stream_mode="values", config = config):
            message = s["messages"][-1]
            out = pretty_print(message)
            log.append(out)
        res = self.output_parser.invoke(s["messages"][-1].content)
        result = {
            'likelihood_h1': res.likelihood_h1,
            'likelihood_h0': res.likelihood_h0
        }
        return result


class falsification_test_proposal_agent:
    def __init__(self, data, llm = 'claude-3-5-sonnet-20241022', domain = "biology", port=None, api_key="EMPTY", base_url=None):
        self.data = data
        self.llm = get_llm(llm, port=port, api_key=api_key, base_url=base_url)
        self.domain = domain
        self.existing_tests = []
        self.failed_tests = []
        
        self.system_prompt = ChatPromptTemplate.from_messages([("system", get_test_proposal_agent_system_prompt(self.domain)), ("human", "{input}")])
        self.chain = self.system_prompt | self.llm.with_structured_output(test_specification)
        self.output_parser = self.llm.with_structured_output(test_specification)

    def go(self, main_hypothesis, test_results=None, log=None):
        if not test_results:
            test_results = self.existing_tests
        prompt_modifier = get_test_proposal_agent_user_prompt(self.domain, main_hypothesis, self.data, test_results, self.failed_tests)

        #print(prompt_modifier)
        self.app = create_react_agent(self.llm, [])

        config = {"recursion_limit": 500}
        inputs = {"messages": [("user", prompt_modifier)]}
        for s in self.app.stream(inputs, stream_mode="values", config = config):
            message = s["messages"][-1]
            out = pretty_print(message)
            pattern = r"={32}\x1b\[1m (Ai|Human) Message \x1b\[0m={32}"
            clean_out = re.sub(pattern, '', out)
            log['designer'].append(clean_out)
        
        for _ in range(10):
            # retry when output_parser fails
            res = self.output_parser.invoke(s["messages"][-1].content)
            if res:
                break
        
        question = "Main hypothesis: {main_hypothesis} \n Falsification Test name: {test_name} \n Falsification Test description: {test_description} \n Falsification Test Null sub-hypothesis: {null_hypothesis} \n Falsification Test Alternate sub-hypothesis: {alternate_hypothesis}".format(main_hypothesis = main_hypothesis, test_name = res.test_name, test_description=res.test_description, null_hypothesis=res.null_hypothesis, alternate_hypothesis=res.alternate_hypothesis)
        return question

    def add_to_existing_tests(self, test):
        self.existing_tests.append(test)

    def add_to_failed_tests(self, test):
        self.failed_tests.append(test)

class SequentialFalsificationTest:
    def __init__(self, llm = 'claude-3-5-sonnet-20241022', is_local=False, port=None, api_key="EMPTY", base_url=None):
        if is_local and base_url is None:
            assert port is not None, "A server port must be provided when using a locally served model."
        self.port = port
        self.api_key = api_key
        self.base_url = base_url
        self.llm_use = llm
        self.llm = get_llm(llm, port=self.port, api_key=self.api_key, base_url=self.base_url)
        self.output_parser = self.llm.with_structured_output(OutputSpecification)
        self.num_of_tests = 0
        self.res = False
        self.res_stat = None

        self.proposal_relevance_checker_prompt = ChatPromptTemplate.from_messages(
            [("system", get_relevance_prompt()),
            ("placeholder", "{messages}"),
            ]
        )
        self.proposal_relevance_checker = self.proposal_relevance_checker_prompt | self.llm.with_structured_output(relevance_subhypothesis)
        
        self.log = {
            'designer': [],
            'executor': [],
            'relevance_checker': [],
            'summarizer': [],
            'sequential_testing': []
        }

    def complete_decision(self, reason):
        """Freeze a terminal statistical outcome before any reporting model call."""
        if self.decision_snapshot is not None:
            return
        statistic = None if self.res_stat is None else float(self.res_stat)
        if statistic is not None and not np.isfinite(statistic):
            statistic = str(statistic)
        self.decision_snapshot = dict(
            completed=True, stop_reason=reason, alpha=self.alpha,
            aggregate_test=self.aggregate_test,
            rejected=bool(self.res) if self.tracked_stat else None,
            statistic=statistic, p_values=list(self.tracked_stat),
            tracked_tests=list(self.tracked_tests))
        if self.decision_path is not None:
            from pathlib import Path
            path = Path(self.decision_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(self.decision_snapshot, indent=2, allow_nan=False), encoding='utf-8')
            temporary.replace(path)

    def summarize(self):
        to_print = [get_msg_title_repr("Summarizer", bold=is_interactive_env())]
        print(to_print[0])
        prompt_modifier = get_summarizer_system_prompt()
        
        prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", prompt_modifier),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )

        test_results = '\n'.join([f"------- Round {i+1} ------- \n Falsification Test: {self.tracked_tests[i]} \n test statistics: {self.tracked_stat[i]}" for i in range(len(self.tracked_tests))])

        if self.aggregate_test == 'LLM_approx':
            res = f"Cumulative Estimated Likelihood: {self.res_stat}"
        elif self.aggregate_test == 'Fisher':
            res = f"Fisher's Method current combined p-value: {self.res_stat}"
        elif self.aggregate_test == 'E-value':
            res = f"E-value current combined e-value using kappa p-to-e calibrator: {self.res_stat}"
        elif self.aggregate_test == 'E-value_integral':
            res = f"E-value current combined e-value using integral p-to-e calibrator: {self.res_stat}"

        res_log = "sufficient evidence - PASS" if self.res else "insufficient evidence - CONTINUE"
        test_results += f"\n\n Sequential testing result: {res_log} with statistics {res} \n Number of total tests done: {self.num_of_tests}"

        agent_executor = create_react_agent(self.llm, [], messages_modifier=prompt)

        config = {"recursion_limit": 500}
        inputs = {"messages": [("user", test_results)]}
        for response in agent_executor.stream(inputs, stream_mode="values", config = config):
            message = response["messages"][-1]
            out = pretty_print(message, printout = True)
            to_print.append(out)

            pattern = r"={32}\x1b\[1m (Ai|Human) Message \x1b\[0m={32}"
            clean_out = re.sub(pattern, '', out)
            self.log['summarizer'].append(clean_out)

        #self.log.append('\n'.join(to_print))

        return {"messages": [('assistant', response["messages"][-1].content)]}


    def configure(self, data, alpha = 0.1, beta = 0.1, aggregate_test = 'E-value', llm_approx = False,
                    max_num_of_tests = 10, plot_agent_architecture = True,
                    time_limit = 10, max_retry = 10, domain="biology", max_failed_tests = 10, relevance_checker = False, use_react_agent = False, **kwargs):
        self.relevance_checker = relevance_checker
        self.max_num_of_tests = max_num_of_tests
        for name, df in data.table_dict.items():
            globals()[name] = df

        self.aggregate_test = aggregate_test
        self.data_loader = data
        self.data = data.data_desc
        self.alpha = alpha
        self.beta = beta
        self.llm_approx = llm_approx
        self.domain = domain
        self.max_failed_tests = max_failed_tests

        if self.llm_approx:
            self.aggregate_test = 'LLM_approx'
            self.likelihood_estimation_agent = likelihood_estimation_agent(llm = self.llm_use, port=self.port, api_key=self.api_key, base_url=self.base_url)

        if use_react_agent and llm_approx:
            raise ValueError("React Falsitication Test Agent does not yet support llm approx")
        
        if use_react_agent:
            self.test_coding_agent = falsification_test_react_agent(self.data_loader, llm =self.llm_use, max_retry=max_retry, domain=self.domain, port=self.port, api_key=self.api_key, base_url=self.base_url)
        else:
            self.test_coding_agent = falsification_test_coding_agent(self.data, self.llm_use, time_limit = time_limit, max_retry = max_retry, llm_approx = self.llm_approx, domain=self.domain, port=self.port, api_key=self.api_key, base_url=self.base_url)

        self.test_proposal_agent = falsification_test_proposal_agent(self.data, self.llm_use, self.domain, port=self.port, api_key=self.api_key, base_url=self.base_url)

        self.tracked_tests = []
        self.tracked_stat = []
        self.test_audit = []
        self.decision_snapshot = None
        self.decision_path = None

        class State(TypedDict):
            messages: Annotated[list, add_messages]
            cur_test_proposal: str

        def design_falsification_test(state: State):
            test_results = '\n'.join([f"------- Round {i+1} ------- \n Falsification Test: {self.tracked_tests[i]} \n test statistics: {self.tracked_stat[i]}" for i in range(len(self.tracked_tests))]) if len(self.tracked_tests) > 0 else "No Implemented Falsification Test Yet."
            if self.relevance_checker:
                for i in range(self.max_failed_tests):
                    proposal = self.test_proposal_agent.go(self.main_hypothesis, test_results, self.log)
                    proposal_check = self.proposal_relevance_checker.invoke({ "messages": [("user", f"Subhypothesis: {proposal}; Main hypothesis: {self.main_hypothesis}")]}).dict()
                    if float(proposal_check['relevance_score']) < 0.8:
                        self.test_proposal_agent.add_to_failed_tests(proposal)
                        self.test_audit.append(dict(kind="relevance", status="rejected",
                                                    proposal=proposal, check=proposal_check))
                        print(f"Proposed falsification test is not relevant enough to the main hypothesis! \n Proposal: \n{proposal} \nRelevance score: {proposal_check['relevance_score']} \nReasoning: {proposal_check['relevance_reasoning']}")
                        self.log['relevance_checker'].append(f"Proposed falsification test is not relevant enough to the main hypothesis! \n Proposal: \n{proposal} \nRelevance score: {proposal_check['relevance_score']} \nReasoning: {proposal_check['relevance_reasoning']}")
                    else:
                        print(f"Proposed falsification test passes relevance check: \n Proposal: {proposal} \nRelevance score {proposal_check['relevance_score']} \nReasoning: {proposal_check['relevance_reasoning']}")
                        self.log['relevance_checker'].append(f"Proposed falsification test passes relevance check: \n Proposal: {proposal} \nRelevance score {proposal_check['relevance_score']} \nReasoning: {proposal_check['relevance_reasoning']}")
                        return {"cur_test_proposal": proposal, "messages": [('assistant', "Proposed falsification test: " + proposal)]}
            else:
                proposal = self.test_proposal_agent.go(self.main_hypothesis, test_results)
                return {"cur_test_proposal": proposal, "messages": [('assistant', "Proposed falsification test: " + proposal)]}

        def implement_falsification_test(state: State):
            event = dict(kind="implementation", status="started", proposal=state["cur_test_proposal"])
            self.test_audit.append(event)
            try:
                out = self.test_coding_agent.go(state["cur_test_proposal"], self.log)
            except Exception as exc:
                event.update(status="failed", failure_reason=type(exc).__name__, error=str(exc),
                             attempts=getattr(self.test_coding_agent, 'last_attempt_audit', []))
                raise
            event.update(attempts=out.get('attempts', []))

            if out['status'] == "Failed test":
                event.update(status="failed", failure_reason=out.get('failure_reason', 'implementation_failed'))
                self.implementation_success_status = False
                self.test_proposal_agent.add_to_failed_tests(state["cur_test_proposal"])
                return {"messages": [('assistant', f"Failed to implement test: {state['cur_test_proposal']}")]}
            else:
                self.implementation_success_status = True
                self.test_proposal_agent.add_to_existing_tests(state["cur_test_proposal"])
                self.tracked_tests.append(state["cur_test_proposal"])

                if self.llm_approx:
                    evidence = out['captured_output']
                    print(get_msg_title_repr("Likelihood ratio estimation agent", bold=is_interactive_env()))
                    out = self.likelihood_estimation_agent.go(self.main_hypothesis, state["cur_test_proposal"], evidence)

                    likelihood_h1 = float(out['likelihood_h1'])
                    likelihood_h0 = float(out['likelihood_h0'])
                    self.tracked_stat.append(likelihood_h1/likelihood_h0)
                    event.update(status="success", likelihood_ratio=float(likelihood_h1/likelihood_h0))
                    return {"messages": [('assistant', f"Falsification test: {state['cur_test_proposal']} \n likelihood under H1: {likelihood_h1} \n likelihood under H0: {likelihood_h0} \n likelihood ratio: {likelihood_h1/likelihood_h0}")]}
                else:
                    self.tracked_stat.append(float(out['p_val']))
                    event.update(status="success", p_value=float(out['p_val']),
                                 fallback_reason=out.get('fallback_reason'))
                    return {"messages": [('assistant', f"Falsification test: {state['cur_test_proposal']} \n p-value: {out['p_val']}")]}

        def sequential_testing(state: State):
            to_print = [get_msg_title_repr("Sequential Testing", bold=is_interactive_env())]
            print(to_print[0])
            if self.aggregate_test == 'Fisher':
                self.res, self.res_stat = fishers_method(self.tracked_stat, alpha=self.alpha)
            elif self.aggregate_test == 'LLM_approx':
                self.res, self.res_stat = likelihood_ratio_e_value(self.tracked_stat, alpha=self.alpha)
            elif self.aggregate_test == 'E-value':
                self.res, self.res_stat = e_value_kappa_calibrator(self.tracked_stat, alpha=self.alpha)
            elif self.aggregate_test == 'E-value_integral':
                self.res, self.res_stat = e_value_integral_calibrator(self.tracked_stat, alpha=self.alpha)
            self.num_of_tests += 1
            res_log = "sufficient evidence - PASS" if self.res else "insufficient evidence - CONTINUE"
            if self.llm_approx:
                output = f"List of likelihood ratios: {self.tracked_stat} \n Summarized sequential statistics: {self.res_stat} \n Sequential test result: {res_log}"
            else:
                output = f"List of p-values: {self.tracked_stat} \n Summarized sequential statistics: {self.res_stat} \n Sequential test result: {res_log}"
            print(output)
            self.log['sequential_testing'].append(output)
            return {"messages": [('assistant', output)]}

        def implementation_status(state: State) -> Literal["sequential_testing", "design_falsification_test"]:
            to_print = [(get_msg_title_repr(f"Falsification test implementation successful? {self.implementation_success_status}", bold=is_interactive_env()))]
            print(to_print[0])
            self.log['sequential_testing'].append(f"Falsification test implementation successful? {self.implementation_success_status}")
            if self.implementation_success_status:
                return "sequential_testing"
            else:
                return "design_falsification_test"

        def test_decision(state: State) -> Literal["design_falsification_test", "summarizer"]:
            res_log = "sufficient evidence - PASS" if self.res else "insufficient evidence - CONTINUE"
            to_print = [(get_msg_title_repr(f"Testing decision is {res_log}", bold=is_interactive_env()))]
            print(to_print[0])
            self.log['sequential_testing'].append(f"Testing decision is {res_log}")
            if self.res:
                return "summarizer"
            else:
                return "design_falsification_test"
            
            # if self.res == "sufficient evidence - PASS":
            #     return "summarizer"
            # elif self.res == 'sufficient evidence for null hypothesis - FAIL':
            #     return "summarizer"
            # else:
            #     return "design_falsification_test"
        
        def summarizer(state: State):
            self.complete_decision("rejected")
            return self.summarize()

        
        graph_builder = StateGraph(State)
        graph_builder.add_node("design_falsification_test", design_falsification_test)
        graph_builder.add_node("implement_falsification_test", implement_falsification_test)
        graph_builder.add_node("sequential_testing", sequential_testing)
        graph_builder.add_node("summarizer", summarizer)        
        
        graph_builder.add_edge(START, "design_falsification_test")
        graph_builder.add_edge("design_falsification_test", "implement_falsification_test")
        graph_builder.add_conditional_edges("implement_falsification_test", implementation_status)
        graph_builder.add_conditional_edges("sequential_testing", test_decision)
        graph_builder.add_edge('summarizer', END)

        self.graph = graph_builder.compile()
        

    def go(self, prompt):
        self.log = {
            'designer': [],
            'executor': [],
            'relevance_checker': [],
            'summarizer': [],
            'sequential_testing': []
        }
        self.main_hypothesis = prompt
        config = {"recursion_limit": 500}
        for s in self.graph.stream({"messages": ("user", prompt)}, stream_mode="values", config = config):
            message = s["messages"][-1]
            out = message.content
            if self.num_of_tests + 1 > self.max_num_of_tests or self.max_failed_tests <= len(self.test_proposal_agent.failed_tests):
                print('Surpassing the maximum number of falsification tests, stopped and summarizing...')
                self.log['summarizer'].append('Surpassing the maximum number of falsification tests, stopped and summarizing...')
                reason = ("rejected" if self.res else "max_tests" if self.num_of_tests >= self.max_num_of_tests
                          else "max_failed_tests")
                self.complete_decision(reason)
                out = self.summarize()['messages'][0][1]
                self.log['summarizer'].append(out)
                break

        result = self.output_parser.invoke(out)
        # result.conclusion = self.res
        return self.log, out, result.dict()
