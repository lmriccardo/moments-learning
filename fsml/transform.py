from multiprocessing import Pool
import fsml.utils as utils
from typing import List
import libsbml
import os


def sbml_to_raterules(document: libsbml.SBMLDocument) -> libsbml.SBMLDocument:
    """
    Convert the SBML model by replacing all the reactions with
    the corresponding Rate Rules, i.e. ODE. In this way we also
    have that all the parameters local to reactions, now they
    become global to all the model. 

    :param document: a handler to the SBML document
    :return:
    """
    try:

        conv_properties = libsbml.ConversionProperties()
        conv_properties.addOption("replaceReactions")
        utils.handle_sbml_errors(document, document.convert(conv_properties))
        return document
    
    except ValueError:
        exit(utils.Errors.CONVERSION_ERROR)


def add_amount_species(document: libsbml.SBMLDocument) -> None:
    """
    Given a :class:`libsbml.SBMLDocument` object that describe
    the current loaded SBML model, add to the species list a
    number of new species such that each of the new specie 
    represent the amount (or particle number) value of the already
    existing species. That is, given a model with N species, we
    add N more species such that given an existing specie `s_i`
    the new one will be `s_i_amount`.

    Then for each of the N new species, we need to add a new 
    assignment rules that map the new specie to the product
    `s_i / compartment`.

    :param document: A handle to the SBML Document
    :return:
    """
    try:
        # First take the corresponding model of the document
        sbml_model: libsbml.Model = document.getModel()

        # Before adding the new species we need to add
        # a new compartment with size 1 to be mapped
        # to all the newly added species
        new_compartment_name = "amount_specie_compartment"
        new_compartment: libsbml.Compartment = sbml_model.createCompartment()
        utils.handle_sbml_errors(document, new_compartment.setName(new_compartment_name))
        utils.handle_sbml_errors(document, new_compartment.setId(new_compartment_name))
        utils.handle_sbml_errors(document, new_compartment.setSize(1.0))

        # Iterate all the species
        number_of_species = sbml_model.getNumSpecies()
        for current_specie in range(number_of_species):
            # Craft the new name and create the new specie
            current_specie_obj: libsbml.Species = sbml_model.getSpecies(current_specie)
            current_specie_name = current_specie_obj.getId()
            current_specie_amount_name = f"{current_specie_name}_amount"
            current_specie_amount: libsbml.Species = sbml_model.createSpecies()
            utils.handle_sbml_errors(document, current_specie_amount.setName(current_specie_amount_name))
            utils.handle_sbml_errors(document, current_specie_amount.setId(current_specie_amount_name))
            utils.handle_sbml_errors(document, current_specie_amount.setCompartment(new_compartment_name))
            utils.handle_sbml_errors(document, current_specie_amount.setHasOnlySubstanceUnits(True))
            utils.handle_sbml_errors(document, current_specie_amount.setConstant(False))

            # We need to set also the initial amount to the new specie
            initial_amount = None

            if current_specie_obj.isSetInitialConcentration:
                initial_amount = current_specie_obj.getInitialConcentration()
                compartment_name = current_specie_obj.getCompartment()
                compartment_obj = sbml_model.getElementBySId(compartment_name)
                initial_amount = initial_amount / compartment_obj.getSize()

            if current_specie_obj.isSetInitialAmount:
                initial_amount = current_specie_obj.getInitialAmount()
            
            utils.handle_sbml_errors(document, current_specie_amount.setInitialAmount(initial_amount))

            # Then get the compartment of the current specie
            current_compartment_name = current_specie_obj.getCompartment()

            # Now we need to add the corresponding assignment rule
            mathXML = "<math xmlns=\"http://www.w3.org/1998/Math/MathML\">" + \
                    "    <apply>"                                         + \
                    "        <divide/>"                                   + \
                    f"        <ci>{current_specie_name}</ci>"              + \
                    f"        <ci>{current_compartment_name}</ci>"         + \
                    "    </apply>"                                        + \
                    "</math>"
            
            mathml_astnode = libsbml.readMathMLFromString(mathXML)
            amount_ass_rule: libsbml.AssignmentRule = sbml_model.createAssignmentRule()
            utils.handle_sbml_errors(document, amount_ass_rule.setVariable(current_specie_amount_name))
            utils.handle_sbml_errors(document, amount_ass_rule.setMath(mathml_astnode))
    
    except ValueError:
        exit(utils.Errors.CONVERSION_ERROR)


def transform(sbmlfile: str, outputfile: str) -> None:
    """
    Load the model, transform the model and finally save the modified model.

    :param sbmlfile:   the absolute path to the SBML model to load
    :param outputfile: the absolute path to the file where to store the new SBML
    :return:
    """
    # Obtain the corresponding document
    document = utils.load_sbml(sbmlfile)

    # Add amount species for all the species
    add_amount_species(document)

    # Convert all reaction to rate rules
    document = sbml_to_raterules(document)

    # Get the handler to the SBML Model
    model = document.getModel()

    # Save the new model
    utils.save_model(model, outputfile)


def transform_model(sbml_path: str) -> str:
    """
    Transform a SBML model using the transform function

    :param sbml_path: the path to the SBML model
    :return: the path where the transformed model has been saved
    """
    # Obtain the output path
    path, extension = sbml_path.split(".")
    output_path = f"{path}_output.{extension}"

    # Call the transform function to perform the transformation
    transform(sbml_path, output_path)

    return output_path


def convert_one(prefix_path: str, model_id: int) -> str:
    """
    Convert one model speciefied by the model_id input parameter

    :param args: a List of arguments for the multiprocessing map function
        This parameter must contains only three elements:
            - prefix_path: the path where to save the model
            - model_id:    the ID of the model in the BioModels database

    :return: the path where the transformed model has been saved
    """
    # Download the model
    model_path = utils.download_model(prefix_path, model_id + 1)
    print(f"({model_id}) [*] Dowloaded file {model_path}")

    # Transform the model
    trans_model_path = transform_model(model_path)
    print(f"({model_id}) [*] Transformed model file {trans_model_path}")

    return trans_model_path


def convert_models(prefix_path: str, nmodels: int) -> List[str]:
    """
    Download and converts a bunch of models

    :param prefix_path: the path where to save the downloaded models
    :param nmodels:     the number of models to download and convert
    :return: a list containing all the paths to the transformed models
    """
    output_paths = []
    cpu_count = os.cpu_count()
    for (minc, maxc) in utils.get_ranges(nmodels, cpu_count):
        with Pool(cpu_count) as pool:
            args = list(map(lambda x: (prefix_path, x), range(minc, maxc)))
            trans_model_paths = pool.map(convert_one, args)
            output_paths += trans_model_paths
    
    return output_paths


if __name__ == "__main__":
    # One conversion example
    import os.path as opath
    prefix_path = opath.join(os.getcwd(), "tests")
    model_id = 0

    convert_one(prefix_path, model_id)