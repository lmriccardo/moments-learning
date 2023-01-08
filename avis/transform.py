from typing import Dict, Tuple, List
import libsbml
import argparse
import os


VERBOSE = False


class Errors:
    FILE_NOT_EXISTS = 1
    XML_READ_ERROR  = 2


def print_errors(doc: libsbml.SBMLDocument, nerrors: int) -> None:
    """
    Print all the errors for a given SBML document

    :param doc:     A handle to the SBML document
    :param nerrors: The number of errors to be printed
    """
    for i in range(0, nerrors):
        format_string = "[line=%d] (%d) <category: %s, severity: %s> %s"
        error: libsbml.XMLError = doc.getError(i)
        error_line = error.getLine()
        error_id = error.getErrorId()
        error_category = error.getCategory()
        error_severity = error.getSeverity()
        error_message = error.getMessage()

        print(format_string % (
            error_line, error_id, error_category,
            error_severity, error_message
        ))

    exit(Errors.XML_READ_ERROR)


def load_sbml(sbmlpath: str) -> libsbml.SBMLDocument:
    """
    Given an SBML file with the absolute path load the file as an SBML document

    :param sbmlpath: the absolute path to the SBML file
    :return: a handler to the SBMLDocument given by the SBML file
    """
    # Read the SBML and obtain the Document
    reader = libsbml.SBMLReader()
    document = reader.readSBML(sbmlpath)
    
    # Check if there are errors after reading and in case print those errors
    if document.getNumErrors() > 0:
        print_errors(document, document.getNumErrors())

    print("--- SBML Loaded without any error")

    return document


def print_species(species: Dict[str, Tuple[str]]) -> None:
    """
    Print all the species parsed from the model

    :param species: the dictionary of all species
    """
    # Obtain the name with the maximum length
    max_len_name = max(map(lambda x: len(x), species.keys()))

    # Iterate and print
    for idx, (specie_name, (specie_id, _)) in enumerate(species.items()):
        spaces = " " * ((max_len_name - len(specie_name)) + 1)
        print("N. %02d) Specie Name: %s%s| Specie Id: %s" % (
            idx + 1, specie_name, spaces, specie_id
        ))


def get_species(model: libsbml.Model) -> Dict[str, Tuple[str, str]]:
    """
    Obtain all the species names as long as their IDs

    :param model: A handler to the SBML Model
    :return: a dictionary with keys specie's name and value specie's ID
    """
    species = dict()

    # Obtain the total number of species in the model
    print("--- Loading all the species of the SBML Model")
    num_species = model.getNumSpecies()
    for i in range(0, num_species):

        # Take the specie at position i
        current_specie: libsbml.Species = model.getSpecies(i)

        # Get the name and the ID and fill the dictionary
        species[current_specie.getName()] = (current_specie.getId(), current_specie.getCompartment())

    if VERBOSE:
        print("--- All the species have been loaded correctly")
        print_species(species)

    return species


def add_output_species(model: libsbml.Model, species: Dict[str, Tuple[str]]) -> List[libsbml.Species]:
    """
    For each specie inside a SBML model, add a corresponding output specie
    which output value will be the normalization of all the values that
    the original specie get through all the simulation process.

    Each new specie will belong to the same compartment of the original specie,
    will have constant attribute False and boundaryCondition attribute True.

    :param model:   a handler to the SBML Model object
    :param species: the dictionary containing all the species of the model
    :return: a list containing all the new SBML Species created
    """
    print("--- Adding output species to the SBML model")
    new_species_list = []
    for idx, (specie_name, (specie_id, specie_compartment)) in enumerate(species.items()):
        # Set new name, new id and new compartment for the new specie
        new_name = specie_name + "_output"
        new_id   = specie_id + "_output"
        new_comp = specie_compartment

        # Create a new empty specie with level 3 version 2
        new_specie = model.createSpecies()

        # Set the new attributes
        new_specie.setName(new_name)
        new_specie.setId(new_id)
        new_specie.setCompartment(new_comp)
        new_specie.setConstant(False)
        new_specie.setBoundaryCondition(True)

        if VERBOSE:
            print("Created new specie N. %02d) " % (idx + 1), new_specie)

        new_species_list.append(new_specie)

    return new_species_list


def add_bparameter(model: libsbml.Model, bvalue: float) -> libsbml.Parameter:
    """
    Add a new constant parameter called "normalization_bvalue" to the Model

    :param model:  a handle to the SBML model
    :param bvalue: the value of the new parameter
    :return: a handle to the newly created parameter
    """
    print("--- Adding a new parameter 'normalization_bvalue' with value %f" % bvalue)
    bparameter: libsbml.Parameter = model.createParameter()
    bparameter.setName("normalization_bvalue")
    bparameter.setId("bparameter")
    bparameter.setValue(bvalue)
    bparameter.setConstant(True)

    if VERBOSE:
        print(bparameter)

    return bparameter

    
def add_normalization_function(model: libsbml.Model) -> libsbml.FunctionDefinition:
    """
    Add a new function for the model that computes the normalization
    of the state (specie) values at each step of the simulation. This
    is used to compute the output value for the new output variable
    (specie) previously added to the model. This function takes as
    input a specie, and a parameter and compute

            Y = (1 + (1 - exp(-b * z)) / (1 + exp(-b * z))) / 2

    where 'p' is the parameter and 'z' is the state.

    :param model: a handle to the SBML model
    :return: the handle to tne newly created function
    """
    print("--- Adding the normalization function to the Model")
    nfunction: libsbml.FunctionDefinition = model.createFunctionDefinition()
    nfunction.setId("nfunction")
    nfunction.setName("State Normalization Function")

    # Create the Math for the function definition
    function_def_mathml = "<math xmlns=\"http://www.w3.org/1998/Math/MathML\">" + \
                          "    <lambda>"                                        + \
                          "        <bvar><ci>z</ci></bvar>"                     + \
                          "        <bvar><ci>b</ci></bvar>"                     + \
                          "        <apply>"                                     + \
                          "            <divide/>"                               + \
                          "            <apply>"                                 + \
                          "                <plus/>"                             + \
                          "                <cn>1</cn>"                          + \
                          "                <apply>"                             + \
                          "                    <divide/>"                       + \
                          "                    <apply>"                         + \
                          "                        <minus/>"                    + \
                          "                        <cn>1</cn>"                  + \
                          "                        <apply>"                     + \
                          "                            <exp/>"                  + \
                          "                            <apply>"                 + \
                          "                                <times/>"            + \
                          "                                <apply>"             + \
                          "                                    <minus/>"        + \
                          "                                    <ci>b</ci>"      + \
                          "                                </apply>"            + \
                          "                                <ci>z</ci>"          + \
                          "                            </apply>"                + \
                          "                        </apply>"                    + \
                          "                    </apply>"                        + \
                          "                    <apply>"                         + \
                          "                        <plus/>"                     + \
                          "                        <cn>1</cn>"                  + \
                          "                        <apply>"                     + \
                          "                            <exp/>"                  + \
                          "                            <apply>"                 + \
                          "                                <times/>"            + \
                          "                                <apply>"             + \
                          "                                    <minus/>"        + \
                          "                                    <ci>b</ci>"      + \
                          "                                </apply>"            + \
                          "                                <ci>z</ci>"          + \
                          "                            </apply>"                + \
                          "                        </apply>"                    + \
                          "                    </apply>"                        + \
                          "                </apply>"                            + \
                          "            </apply>"                                + \
                          "            <cn>2</cn>"                              + \
                          "        </apply>"                                    + \
                          "    </lambda>"                                       + \
                          "</math>"

    # Obtain the ASTNode that corresponds to the mathml formula
    mathml_astnode = libsbml.readMathMLFromString(function_def_mathml)

    # Set the formula for the function
    nfunction.setMath(mathml_astnode)

    if VERBOSE:
        print(nfunction)

    print(libsbml.formulaToL3String(nfunction.getMath()))

    return nfunction


def add_rules(model     : libsbml.Model, 
              species   : Dict[str, Tuple[str, str]], 
              ospecies  : List[libsbml.Species], 
              bparam    : libsbml.Parameter,
              nfunction : libsbml.FunctionDefinition) -> None:
    """
    For each species add an assignment rule in the model that associate the
    corresponding new output specie with the normalization of the value which
    that specie has in any time of the simulation. That is, for each specie
    add a rule like: specie_output = norm_function(specie, bparam).

    :param model:     a handle to the SBML Model
    :param species:   a dictionary containing all the original specie of the model
    :param ospecies:  a list of all the new output species
    :param bparam:    the normalization parameter
    :param nfunction: a handle to the normalization function definition
    :return:
    """
    print("--- Adding Assignment Rules for each species")
    for idx, (_, (specie_id, _)) in enumerate(species.items()):
        # Obtain the corresponding output specie
        ospecie = ospecies[idx]

        # Add a new assignment rule in the model
        assignment_rule: libsbml.AssignmentRule = model.createAssignmentRule()
        assignment_rule.setVariable(ospecie.getId())
        assignment_rule.setId(f"{ospecie.getId()}_{specie_id}_assrule")
        assignment_rule.setName(f"Normalization {specie_id} to {ospecie.getId()}")

        # Create and assign a new ASTNode
        specie_node = libsbml.ASTNode(libsbml.AST_NAME)
        bparam_node = libsbml.ASTNode(libsbml.AST_NAME)
        function_node = libsbml.ASTNode(libsbml.AST_FUNCTION)

        specie_node.setName(specie_id)
        bparam_node.setName(bparam.getId())
        function_node.setName(nfunction.getId())

        function_node.addChild(specie_node)
        function_node.addChild(bparam_node)

        assignment_rule.setMath(function_node)


def save_model(model: libsbml.Model, path: str) -> None:
    """
    Write a SBML Model inside a file

    :param model: a handle to the SBML model
    :param path:  a string representing the output path
    """
    print(f"--- Saving the new modified model to {path}")
    writer = libsbml.SBMLWriter()
    document = model.getSBMLDocument()
    writer.writeSBMLToFile(document, path)


def transform(sbmlfile: str, outputfile: str, bvalue: float) -> None:
    """
    Load the model, transform the model and finally save the modified model.
    The transformation involves: adding new output species, which will have
    the normalized value for each respective specie through the entire
    simulation; adding a new parameter, the normalization parameter; adding
    a new funciton, the normalization function that takes as input a specie
    and the normalization parameter and compute the normalized value; adding
    for each specie an assignment rule that maps the current value of that
    specie with the output specie. 

    :param sbmlfile:   the absolute path to the SBML model to load
    :param outputfile: the absolute path to the file where to store the new SBML
    :param bvalue:     the input value for the normalization parameter
    :return:
    """
    # Obtain the corresponding document
    document = load_sbml(sbmlfile)
    model = document.getModel()

    # Obtain all the species and add output species to the model
    species = get_species(model)
    output_species = add_output_species(model, species)

    # Add the new parameter
    bparameter = add_bparameter(model, bvalue)

    # Add the new function to the model
    nfunction = add_normalization_function(model)

    # Add new rules
    add_rules(model, species, output_species, bparameter, nfunction)

    # Save the new model
    save_model(model, outputfile)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--sbml",    help="The path to the SBML file", required=True)
    parser.add_argument("-v", "--verbose", help="Set the verbosity or not",  action="store_true", default=False)
    parser.add_argument("-b", "--bvalue",  help="The initial value of the parameter b", default=0.01, type=float)
    parser.add_argument("-o", "--output",  help="The path to save the modified model")
    args = parser.parse_args()

    # Set the verbosity
    global VERBOSE
    VERBOSE = args.verbose

    # Obtain the b parameter value
    bvalue = args.bvalue
    if not bvalue: bvalue = 0.01

    # Obtain the SBML file (relative or absolute path)
    sbmlfile = args.sbml

    # In any case check if the file exists or not
    sbml_abspath = os.path.abspath(sbmlfile)
    if not os.path.exists(sbml_abspath):
        print("ERROR: %s file does not exists" % sbmlfile)
        exit(Errors.FILE_NOT_EXISTS)

    # Obtain the output path
    output_path = args.output
    if not output_path:
        splitted_path = sbml_abspath.split(".")
        output_path = splitted_path[0] + "_norm." + splitted_path[1]
        
    output_path = os.path.abspath(output_path)

    transform(sbml_abspath, output_path, bvalue)


if __name__ == "__main__":
    main()